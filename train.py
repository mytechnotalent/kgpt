import os
import math
import numpy as np
import torch
import torch.nn as nn
import warnings
from model import BigramLanguageModel, device, enc, block_size

# ignore cuda warnings
warnings.filterwarnings("ignore")

# the batch_size parameter determines how many independent sequences will be
# processed in parallel during training and increasing the batch size allows
# for more efficient computation and parallelization but may require more memory
# and a larger batch size can also provide a more stable gradient estimation
# but might lead to slower convergence or generalization issues
batch_size = 12  # how many independent sequences will we process in parallel?
# the max_iters parameter represents the maximum number of iterations or steps during the
# training process and it determines how many times the model will update its parameters
# based on the training data and increasing max_iters allows for more training iterations,
# potentially leading to better model performance, however, it may also increase the training
# time and the risk of overfitting if the model starts memorizing the training data
max_iters = 600000
# the eval_interval parameter specifies the frequency at which the model's performance
# is evaluated on the training and validation sets and it determines how often the loss
# values are printed or logged during training and a smaller eval_interval provides more
# frequent updates on the model's progress but can increase the computational overhead
# and adjusting this parameter depends on the desired level of monitoring and the trade-off
# between evaluation frequency and training efficiency
eval_interval = 2000
# the learning_rate parameter controls the step size at each iteration during the model's
# parameter update using gradient descent optimization and it determines how much the model's
# parameters are adjusted based on the computed gradients and a higher learning rate allows
# for larger updates, potentially leading to faster convergence, however, using a very high
# learning rate can cause the optimization process to become unstable or prevent
# convergence, on the other hand, a lower learning rate may require more iterations for
# convergence but can provide more precise parameter updates
learning_rate = 6e-4
# the eval_iters parameter determines the number of iterations used to estimate the loss on the
# training and validation sets during evaluation and it represents the number of iterations used
# to compute the average loss value and a larger eval_iters value provides a more accurate estimation
# of the loss but can increase the evaluation time and adjusting this parameter depends on the
# desired level of accuracy in the loss estimation and the trade-off between evaluation time and accuracy
eval_iters = 200
# the warmup_iters parameter specifies the number of initial training iterations during which
# the learning rate linearly increases from zero to the target value and this warmup phase
# prevents early training instability by starting with small updates before transitioning
# to the full learning rate and is standard practice for large transformer training
warmup_iters = 2000
# the lr_decay_iters parameter specifies the number of iterations over which the learning
# rate cosine-decays from its peak value down to min_lr and this schedule follows the GPT-2
# training recipe and helps the model converge to a lower final loss
lr_decay_iters = 600000
# the min_lr parameter specifies the floor learning rate after cosine decay completes and
# is typically set to one tenth of the peak learning rate following standard transformer
# training practice
min_lr = 6e-5
# the gradient_accumulation_steps parameter allows simulating larger effective batch sizes
# by accumulating gradients over multiple mini-batches before performing a single optimizer
# step and this is essential when GPU memory cannot fit the desired batch size in one pass
gradient_accumulation_steps = 5

# print device either cuda, mps, or cpu
print(device)

# load pre-tokenized binary dataset from the data directory where each file
# contains uint16 token IDs produced by prepare_data.py and memory mapping
# allows efficient random access without loading the entire dataset into RAM
data_dir = os.path.join(os.path.dirname(__file__), "data")
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    """
    Retrieves a batch of data for a given split.

    Args:
        split (str): Specifies the split of the data to retrieve ('train' or 'val').

    Returns:
        tuple: A tuple containing the input data and corresponding target data.
            - x (torch.Tensor): Input data of shape (batch_size, block_size).
            - y (torch.Tensor): Target data of shape (batch_size, block_size).

    Example:
        # Retrieve a training batch
        x_train, y_train = get_batch('train')
    """
    # randomly select starting indices for the batch
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
    Estimates the average loss for a single dataset split.

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
        _, loss = model(X, Y)
        losses[k] = loss.item()
    return losses.mean()


@torch.no_grad()
def estimate_loss():
    """
    Estimates the average loss on the training and validation datasets.

    Returns:
        dict: A dictionary containing the average loss for each dataset split.
            - 'train' (float): Average loss on the training dataset.
            - 'val' (float): Average loss on the validation dataset.

    Example:
        loss_estimation = estimate_loss()
        train_loss = loss_estimation['train']
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        out[split] = _estimate_split_loss(split)
    model.train()
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


# create instance of the BigramLanguageModel class and move to the specified device
model = BigramLanguageModel()
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer with weight decay matching the GPT-2 training configuration
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1
)

# loop iterates over a specified number of iterations (max_iters)
# used for training the model and performing updates on the parameters
for iter in range(max_iters):
    # update the learning rate for this iteration using the warmup and cosine decay schedule
    # which matches the GPT-2 training recipe for stable convergence
    lr = _get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # checks if it's time to evaluate the loss on the training and
    # validation sets and it is determined by the value of eval_interval
    # or if it's the last iteration (iter == max_iters - 1) and
    # the estimate_loss() function is called to compute the losses, and
    # then the losses are printed to provide feedback on the model's performance
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f'step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}'
        )
    # gradient accumulation loop processes multiple micro-batches before performing
    # a single optimizer step which simulates a larger effective batch size of
    # batch_size * gradient_accumulation_steps without requiring extra GPU memory
    for micro_step in range(gradient_accumulation_steps):
        # sample a batch of data (xb) and its corresponding targets (yb)
        xb, yb = get_batch("train")
        # evaluate the loss
        logits, loss = model(xb, yb)
        # scale the loss by the number of accumulation steps so the total gradient
        # magnitude matches what a single large batch would produce
        loss = loss / gradient_accumulation_steps
        # compute the gradients of the loss with respect to the model's parameters
        loss.backward()
    # clip gradients to a maximum norm of 1.0 to prevent training instability
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # update the model's parameters based on the computed gradients
    optimizer.step()
    # set all the gradients to zero to prepare for the next iteration
    optimizer.zero_grad(set_to_none=True)

# save the pretrained model weights to disk for fine-tuning or inference
torch.save(m.state_dict(), "kgpt_pretrained.pt")
print("Model saved to kgpt_pretrained.pt")
