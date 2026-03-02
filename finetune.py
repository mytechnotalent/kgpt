import json
import math
import torch
import torch.nn as nn
import warnings
from model import BigramLanguageModel, device, enc, block_size

# ignore cuda warnings
warnings.filterwarnings("ignore")

# the finetune_iters parameter determines how many gradient update steps to perform
# during the conversational fine-tuning phase and a smaller number is appropriate
# because the pretrained model already has strong language understanding and only
# needs to adapt its output distribution to the conversational format
finetune_iters = 3000
# the finetune_lr parameter is the peak learning rate used during fine-tuning and
# it is set lower than the pretraining learning rate to preserve the pretrained
# knowledge while gently steering the model toward conversational behavior
finetune_lr = 1e-5
# the finetune_warmup parameter specifies the number of iterations for the linear
# learning rate warmup phase at the start of fine-tuning to avoid large initial
# parameter updates that could corrupt pretrained weights
finetune_warmup = 100
# the eval_interval parameter specifies how frequently to print the current training
# loss during fine-tuning so you can monitor convergence progress
eval_interval = 100
# the eval_iters parameter determines the number of batches used to estimate the
# average training loss at each evaluation checkpoint
eval_iters = 20
# the batch_size parameter controls how many conversational examples are packed into
# each training batch during fine-tuning
batch_size = 4
# the dropout parameter is set to a small nonzero value during fine-tuning to
# provide light regularization and prevent overfitting on the small conversational
# dataset unlike pretraining which used dropout of 0.0
dropout = 0.1
# the gradient_accumulation_steps parameter allows simulating a larger effective
# batch size during fine-tuning by accumulating gradients over multiple forward
# and backward passes before updating the model parameters
gradient_accumulation_steps = 4

# use mixed precision (fp16) on cuda to cut activation memory in half
use_amp = device == "cuda"

# print device either cuda, mps, or cpu
print(device)


def _load_training_data(path):
    """
    Loads the conversational training data from a JSON file on disk.

    Args:
        path (str): The file path to the training_data.json file containing
            a list of user/assistant conversation pairs.

    Returns:
        list: A list of dictionaries each containing 'user' and 'assistant' keys
            with their corresponding conversation text strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_conversation(example):
    """
    Formats a single conversation example into a prompt-response string using
    special delimiter tokens that the model learns to associate with turn boundaries.

    Args:
        example (dict): A dictionary with 'user' and 'assistant' keys containing
            the conversation text for each speaker.

    Returns:
        str: A formatted string with user and assistant turns separated by newlines
            and terminated with an end-of-conversation marker.
    """
    user_text = example["user"]
    assistant_text = example["assistant"]
    return f"<|user|>\n{user_text}\n<|assistant|>\n{assistant_text}\n<|end|>\n"


def _tokenize_conversations(conversations):
    """
    Tokenizes all formatted conversation strings into a single flat list of
    GPT-2 BPE token IDs suitable for causal language model training.

    Args:
        conversations (list): A list of conversation dictionaries each containing
            'user' and 'assistant' keys.

    Returns:
        list: A flat list of integer token IDs representing all conversations
            concatenated together with proper formatting delimiters.
    """
    all_tokens = []
    for example in conversations:
        text = _format_conversation(example)
        all_tokens.extend(enc.encode(text))
    return all_tokens


def _build_tensor_dataset(tokens):
    """
    Converts a flat list of token IDs into a single PyTorch long tensor and
    moves it to the target compute device for efficient batch sampling.

    Args:
        tokens (list): A flat list of integer token IDs to convert.

    Returns:
        torch.Tensor: A one-dimensional long tensor of token IDs on the target device.
    """
    return torch.tensor(tokens, dtype=torch.long, device=device)


def _get_finetune_batch(data_tensor):
    """
    Samples a random batch of input-target sequence pairs from the tokenized
    conversation tensor for causal language model fine-tuning.

    Args:
        data_tensor (torch.Tensor): A one-dimensional tensor of token IDs.

    Returns:
        tuple: A tuple of (x, y) tensors each of shape (batch_size, block_size)
            where x is the input sequence and y is the target shifted by one token.
    """
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i : i + block_size] for i in ix])
    y = torch.stack([data_tensor[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def _get_finetune_lr(it):
    """
    Computes the learning rate for a given fine-tuning iteration using linear
    warmup followed by cosine decay to the minimum learning rate.

    Args:
        it (int): The current fine-tuning iteration number.

    Returns:
        float: The computed learning rate value for the given iteration.
    """
    min_lr = finetune_lr * 0.1
    if it < finetune_warmup:
        return finetune_lr * it / finetune_warmup
    if it > finetune_iters:
        return min_lr
    decay_ratio = (it - finetune_warmup) / (finetune_iters - finetune_warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (finetune_lr - min_lr)


def _estimate_finetune_loss(model_ref, data_tensor):
    """
    Estimates the average training loss over a number of random batches from the
    fine-tuning dataset without computing gradients for efficiency.

    Args:
        model_ref (BigramLanguageModel): The model instance to evaluate.
        data_tensor (torch.Tensor): The tokenized fine-tuning data tensor.

    Returns:
        float: The mean loss value averaged over eval_iters random batches.
    """
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        xb, yb = _get_finetune_batch(data_tensor)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model_ref(xb, yb)
        losses[k] = loss.item()
    return losses.mean().item()


# load the pretrained model weights from the checkpoint saved by train.py
model = BigramLanguageModel()
model.load_state_dict(
    torch.load("kgpt_pretrained.pt", map_location=device, weights_only=True)
)
model = model.to(device)
print(
    f"Loaded pretrained model ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)"
)

# load and tokenize the conversational training data from training_data.json
# which contains user/assistant pairs formatted with special delimiter tokens
conversations = _load_training_data("training_data.json")
tokens = _tokenize_conversations(conversations)
data_tensor = _build_tensor_dataset(tokens)
print(f"Fine-tuning on {len(conversations)} conversations ({len(tokens):,} tokens)")

# enable dropout for fine-tuning regularization by updating all dropout modules
# to use the finetune dropout rate instead of the pretraining rate of 0.0
for module in model.modules():
    if isinstance(module, nn.Dropout):
        module.p = dropout

# create the fine-tuning optimizer with lower learning rate and weight decay to
# preserve pretrained knowledge while adapting to conversational patterns
optimizer = torch.optim.AdamW(
    model.parameters(), lr=finetune_lr, betas=(0.9, 0.95), weight_decay=0.01
)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# fine-tuning training loop that iterates for finetune_iters steps with learning
# rate warmup and cosine decay and gradient accumulation for stable convergence
model.train()
for iter in range(finetune_iters):
    # update the learning rate for this iteration using warmup and cosine decay
    lr = _get_finetune_lr(iter)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # periodically estimate and print the fine-tuning loss to monitor convergence
    if iter % eval_interval == 0 or iter == finetune_iters - 1:
        with torch.no_grad():
            model.eval()
            loss_val = _estimate_finetune_loss(model, data_tensor)
            model.train()
        print(f"finetune step {iter}: loss {loss_val:.4f}")
    # gradient accumulation loop to simulate larger effective batch size
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = _get_finetune_batch(data_tensor)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
    # clip gradients to prevent training instability
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # update model parameters and reset gradients
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

# save the fine-tuned model weights to disk for use by inference.py
torch.save(model.state_dict(), "kgpt_finetuned.pt")
print("Fine-tuned model saved to kgpt_finetuned.pt")
