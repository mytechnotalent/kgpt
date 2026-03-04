import json
import math
import torch
import torch.nn as nn
import warnings
from model import GPT, enc, block_size

warnings.filterwarnings("ignore")

# the finetune_iters parameter determines how many gradient update steps to perform
# during the conversational fine-tuning phase using a smaller count since the
# pretrained model already has strong language understanding
finetune_iters = 6000
# the finetune_lr parameter is the peak learning rate during fine-tuning set lower
# than pretraining to preserve pretrained knowledge while adapting to conversations
finetune_lr = 5e-5
# the finetune_warmup parameter specifies iterations for linear learning rate warmup
# to avoid large initial updates that could corrupt pretrained weights
finetune_warmup = 200
# the eval_interval parameter specifies how frequently to print training loss
# during fine-tuning to monitor convergence progress
eval_interval = 200
# the eval_iters parameter determines the number of batches used to estimate
# the average training loss at each evaluation checkpoint
eval_iters = 20
# the batch_size parameter controls how many examples are packed into each
# training batch during fine-tuning
batch_size = 4
# the dropout parameter provides light regularization during fine-tuning to
# prevent overfitting on the small conversational dataset
dropout = 0.1
# the gradient_accumulation_steps parameter simulates a larger effective batch
# size during fine-tuning for more stable gradient estimates
gradient_accumulation_steps = 4

# select the best available compute device prioritizing cuda then mps then cpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# enable automatic mixed precision on cuda to halve activation memory usage
use_amp = device == "cuda"

# print the selected compute device
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


def _tokenize_per_example(conversations):
    """
    Tokenizes each conversation individually into separate token sequences with
    prompt length tracking so that label masking can exclude user prompt tokens
    from the training loss and only train on assistant response tokens.

    Args:
        conversations (list): A list of conversation dictionaries each containing
            'user' and 'assistant' keys.

    Returns:
        list: A list of dictionaries each containing 'tokens' (the full token
            sequence for one conversation) and 'prompt_len' (number of tokens
            in the user prompt portion used to mask non-response positions).
    """
    examples = []
    for ex in conversations:
        full_text = _format_conversation(ex)
        full_tokens = enc.encode(full_text)
        prompt_text = f"<|user|>\n{ex['user']}\n<|assistant|>\n"
        prompt_len = len(enc.encode(prompt_text))
        examples.append({"tokens": full_tokens, "prompt_len": prompt_len})
    return examples


def _get_finetune_batch(examples):
    """
    Samples a random batch of individual conversations and constructs padded
    input-target pairs with label masking so the model only learns to predict
    assistant response tokens and ignores user prompt and padding positions.

    Each conversation is padded to the longest sequence in the batch using the
    eot_token. Target positions corresponding to user prompt tokens and padding
    are set to -100 which is ignored by F.cross_entropy.

    Args:
        examples (list): A list of dictionaries from _tokenize_per_example each
            containing 'tokens' and 'prompt_len' keys.

    Returns:
        tuple: A tuple of (x, y) tensors where x has shape (batch_size, max_len)
            containing input token IDs padded with eot_token, and y has the same
            shape with -100 at prompt and padding positions so only assistant
            response tokens contribute to the training loss.
    """
    indices = torch.randint(len(examples), (batch_size,))
    batch = [examples[i] for i in indices]
    max_seq_len = max(len(b["tokens"]) for b in batch)
    pad_id = enc.eot_token
    x_list = []
    y_list = []
    for b in batch:
        tokens = b["tokens"]
        prompt_len = b["prompt_len"]
        seq_len = len(tokens) - 1
        pad_len = (max_seq_len - 1) - seq_len
        x = tokens[:-1] + [pad_id] * pad_len
        y = [-100] * (prompt_len - 1) + tokens[prompt_len:] + [-100] * pad_len
        x_list.append(torch.tensor(x, dtype=torch.long))
        y_list.append(torch.tensor(y, dtype=torch.long))
    return torch.stack(x_list).to(device), torch.stack(y_list).to(device)


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


def _estimate_finetune_loss(model_ref, examples):
    """
    Estimates the average training loss over a number of random batches from the
    fine-tuning dataset without computing gradients for efficiency.

    Args:
        model_ref (GPT): The model instance to evaluate.
        examples (list): The per-example tokenized dataset from _tokenize_per_example.

    Returns:
        float: The mean loss value averaged over eval_iters random batches.
    """
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        xb, yb = _get_finetune_batch(examples)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model_ref(xb, yb)
        losses[k] = loss.item()
    return losses.mean().item()


# load the pretrained model weights from the checkpoint saved by train.py
model = GPT()
model.load_state_dict(
    torch.load("kgpt_pretrained.pt", map_location=device, weights_only=True)
)
model = model.to(device)
print(
    f"Loaded pretrained model ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)"
)

# tokenize each conversation individually so every training sample is one complete
# conversation with label masking instead of random slices across conversations
conversations = _load_training_data("training_data.json")
examples = _tokenize_per_example(conversations)
total_tokens = sum(len(ex["tokens"]) for ex in examples)
print(f"Fine-tuning on {len(conversations)} conversations ({total_tokens:,} tokens)")

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
            loss_val = _estimate_finetune_loss(model, examples)
            model.train()
        print(f"finetune step {iter}: loss {loss_val:.4f}")
    # gradient accumulation loop to simulate larger effective batch size
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = _get_finetune_batch(examples)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
    # clip gradients to prevent training instability then update parameters
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

# save the fine-tuned model weights to disk for use by inference.py
torch.save(model.state_dict(), "kgpt_finetuned.pt")
print("Fine-tuned model saved to kgpt_finetuned.pt")
