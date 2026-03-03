import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
from model import GPT, enc, block_size

warnings.filterwarnings("ignore")

# detect distributed training by checking the RANK environment variable which
# is set automatically by torchrun when launching multi-GPU training
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    master_process = True

# enable TF32 precision for faster matrix multiplications on Ampere and newer GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# the batch_size parameter determines how many independent sequences will be
# processed in parallel during each micro-step of training
batch_size = 4
# the max_iters parameter represents the total number of training iterations
# determining how many optimizer steps the model performs during pretraining
max_iters = 50000
# the eval_interval parameter specifies the frequency at which the model's
# performance is evaluated on training and validation sets and checkpoints are saved
eval_interval = 2000
# the learning_rate parameter controls the peak step size during optimization
# following the GPT-2 training recipe with warmup and cosine decay
learning_rate = 6e-4
# the eval_iters parameter determines the number of batches used to estimate
# the average loss during each evaluation checkpoint
eval_iters = 200
# the warmup_iters parameter specifies how many iterations the learning rate
# linearly increases from zero to the peak value for training stability
warmup_iters = 2000
# the lr_decay_iters parameter specifies the number of iterations over which
# the learning rate cosine-decays from peak to min_lr
lr_decay_iters = 50000
# the min_lr parameter specifies the floor learning rate after cosine decay
# typically set to one tenth of the peak learning rate
min_lr = 6e-5
# the gradient_accumulation_steps parameter simulates larger effective batch
# sizes by accumulating gradients over multiple micro-batches before updating
gradient_accumulation_steps = 15

# adjust gradient accumulation steps for distributed training so each GPU
# handles an equal share of the effective batch
assert gradient_accumulation_steps % ddp_world_size == 0
gradient_accumulation_steps //= ddp_world_size

# the checkpoint_path specifies the file used for saving and resuming
# training progress across sessions
checkpoint_path = "kgpt_checkpoint.pt"

# enable automatic mixed precision on CUDA to halve activation memory usage
use_amp = "cuda" in str(device)

# set reproducible random seed with offset per rank for different data on each GPU
torch.manual_seed(1337 + ddp_rank)

# print configuration on the master process only
if master_process:
    print(f"Device: {device}")
    if ddp:
        print(f"DDP: world_size={ddp_world_size}")
    print(f"Gradient accumulation steps (per GPU): {gradient_accumulation_steps}")

# load pre-tokenized binary dataset from the data directory where each file
# contains uint16 token IDs produced by prepare_data.py
data_dir = os.path.join(os.path.dirname(__file__), "data")
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    """
    Retrieves a batch of data for a given split by randomly sampling starting
    positions and constructing input-target pairs from the tokenized data.

    Args:
        split (str): Specifies the split of the data to retrieve ('train' or 'val').

    Returns:
        tuple: A tuple containing the input data and corresponding target data.
            - x (torch.Tensor): Input data of shape (batch_size, block_size).
            - y (torch.Tensor): Target data of shape (batch_size, block_size).

    Example:
        x_train, y_train = get_batch('train')
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + block_size + 1]).astype(np.int64))
            for i in ix
        ]
    )
    x, y = x.to(device), y.to(device)
    return x, y


def _estimate_split_loss(split):
    """
    Estimates the average loss for a single dataset split by running multiple
    forward passes without gradient computation for efficiency.

    Args:
        split (str): Specifies the split of the data to evaluate ('train' or 'val').

    Returns:
        torch.Tensor: The mean loss value across eval_iters batches for the given split.

    Example:
        train_loss = _estimate_split_loss('train')
    """
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(split)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = raw_model(X, Y)
        losses[k] = loss.item()
    return losses.mean()


@torch.no_grad()
def estimate_loss():
    """
    Estimates the average loss on the training and validation datasets using
    the unwrapped model to avoid DDP overhead during evaluation.

    Returns:
        dict: A dictionary containing the average loss for each dataset split.
            - 'train' (float): Average loss on the training dataset.
            - 'val' (float): Average loss on the validation dataset.

    Example:
        loss_estimation = estimate_loss()
        train_loss = loss_estimation['train']
    """
    out = {}
    raw_model.eval()
    for split in ["train", "val"]:
        out[split] = _estimate_split_loss(split)
    raw_model.train()
    return out


def _get_lr(it):
    """
    Computes the learning rate for a given iteration using linear warmup followed
    by cosine decay, matching the GPT-2 training schedule.

    Args:
        it (int): The current training iteration number.

    Returns:
        float: The computed learning rate value for the given iteration.
    """
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# create model instance and move to the target compute device
model = GPT().to(device)

# apply torch.compile for potential 2x speedup on supported CUDA hardware
use_compile = hasattr(torch, "compile") and "cuda" in str(device)
if use_compile:
    model = torch.compile(model)

# store reference to the unwrapped model for saving checkpoints and evaluation
# then wrap in DDP for distributed gradient synchronization across GPUs
raw_model = model
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

if master_process:
    print(f"{sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M parameters")

# create a PyTorch optimizer with weight decay on the unwrapped model parameters
optimizer = torch.optim.AdamW(
    raw_model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1
)

# gradient scaler for mixed precision training to prevent underflow in fp16
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# resume from checkpoint if one exists from a previous training session
start_iter = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_iter = checkpoint["iter"] + 1
    if master_process:
        print(f"Resumed from checkpoint at iteration {start_iter}")

# training loop with mixed precision, learning rate scheduling, gradient accumulation,
# and DDP gradient synchronization for multi-GPU efficiency
model.train()
for iter in range(start_iter, max_iters):
    # update learning rate using warmup then cosine decay schedule
    lr = _get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # evaluate loss and save checkpoint at regular intervals on the master process
    if iter % eval_interval == 0 or iter == max_iters - 1:
        if master_process:
            losses = estimate_loss()
            print(
                f'step {iter}: train loss {losses["train"]:.4f}, '
                f'val loss {losses["val"]:.4f}'
            )
            torch.save(
                {
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "iter": iter,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved at iteration {iter}")
    # accumulate gradients over multiple micro-batches before optimizer step
    # only synchronize DDP gradients on the last micro-step for efficiency
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        xb, yb = get_batch("train")
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
    # clip gradients and update model parameters using the gradient scaler
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

# save the pretrained model weights on the master process for fine-tuning
if master_process:
    torch.save(raw_model.state_dict(), "kgpt_pretrained.pt")
    print("Model saved to kgpt_pretrained.pt")

# clean up distributed process group if running in DDP mode
if ddp:
    dist.destroy_process_group()
