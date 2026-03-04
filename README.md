![image](https://github.com/mytechnotalent/kgpt/blob/main/kgpt.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Today's Tutorial [March 4, 2026]
## Lesson 123: x64 Course (Part 3 - Logic Gates)
This tutorial will discuss logic gates.

-> Click [HERE](https://0xinfection.github.io/reversing) to read the FREE ebook.

<br>

# KGPT
A GPT-2-class language model trained from scratch on [OpenWebText](https://huggingface.co/datasets/openwebtext) based on [Zero To Hero](https://karpathy.ai/zero-to-hero.html) utilizing tiktoken with the intent to augment AI Transformer-model education and reverse engineer GPT models from scratch.

The model matches the nanoGPT / GPT-2 small architecture (`n_embd=768`, `n_head=12`, `n_layer=12`, `block_size=1024`, ~124M parameters) with weight tying, fused CausalSelfAttention, scaled residual init, DDP support, and `torch.compile`. It trains on the full OpenWebText dataset. After pretraining the model can be fine-tuned on conversational data to produce a real chatbot.

<br>

## Repository Files
| File                        | Description                                                      |
| --------------------------- | ---------------------------------------------------------------- |
| `model.py`                  | GPT-2 architecture (CausalSelfAttention, Block, GPT)             |
| `train.py`                  | Pretrains the model on OpenWebText (DDP + torch.compile)         |
| `finetune.py`               | Fine-tunes the pretrained model on training_data.json            |
| `inference.py`              | Interactive chatbot with temperature and top-k sampling          |
| `prepare_data.py`           | Downloads OpenWebText and creates tokenized binary files         |
| `generate_training_data.py` | Generates 10,000 diverse Q&A training pairs across 22 categories |
| `training_data.json`        | Conversational dataset (user / assistant pairs)                  |
| `pyproject.toml`            | Project metadata and dependencies                                |
| `kgpt-lite.ipynb`           | Self-contained Kaggle notebook (train + finetune + inference)    |

<br>

## Setup

### 1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -e .
```

### 3. OPTIONAL — install PyTorch with CUDA
Visit [pytorch.org](https://pytorch.org) for your specific configuration. Example:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

### 4. Prepare the dataset
Download, tokenize, and save OpenWebText to `data/train.bin` and `data/val.bin`:
```bash
python prepare_data.py
```
> **Note:** This downloads ~54 GB of text and tokenizes it into ~9 B GPT-2 BPE tokens. The process takes several hours and requires ~60 GB of free disk space.

<br>

## Step 1 — Pretrain
```bash
python train.py
```

The script will:
1. Print the device (`cuda`, `mps`, or `cpu`).
2. Train the transformer for 50,000 iterations with learning-rate warmup and cosine decay, printing loss every 2,000 steps.
3. Save the pretrained weights to `kgpt_pretrained.pt`.

For multi-GPU training via DDP:
```bash
torchrun --nproc_per_node=N train.py
```

> **Kaggle GPU training:** Upload `kgpt-lite.ipynb` to Kaggle with your dataset and enable a T4 GPU. The notebook completes pretraining, fine-tuning, and inference in a single session (~8–9 hours).

<br>

## Step 2 — Fine-tune
```bash
python finetune.py
```

Loads `kgpt_pretrained.pt`, fine-tunes on `training_data.json` for 3 000 iterations with a lower learning rate and light dropout, and saves the result to `kgpt_finetuned.pt`.

<br>

## Step 3 — Chat
```bash
python inference.py
```

Loads `kgpt_finetuned.pt` and starts an interactive chatbot session with temperature sampling, top-k filtering, repetition penalty, and multi-turn conversation history. Type `quit` to exit or `clear` to reset the conversation.

<br>

## Device Support
| Device | Detected When                                    |
| ------ | ------------------------------------------------ |
| `cuda` | NVIDIA GPU with CUDA runtime available           |
| `mps`  | Apple Silicon GPU with Metal Performance Shaders |
| `cpu`  | Fallback when no GPU backend is detected         |

The device is selected automatically at startup using the priority order `cuda > mps > cpu`.

<br>

## Hyperparameters
| Parameter                     | Value  | Purpose                                                 |
| ----------------------------- | ------ | ------------------------------------------------------- |
| `batch_size`                  | 4      | Parallel sequences per micro-batch                      |
| `block_size`                  | 1024   | Maximum context length                                  |
| `max_iters`                   | 50,000 | Total training iterations                               |
| `learning_rate`               | 6e-4   | Peak AdamW step size                                    |
| `warmup_iters`                | 2,000  | Linear LR warmup iterations                             |
| `lr_decay_iters`              | 50,000 | Cosine decay horizon                                    |
| `min_lr`                      | 6e-5   | Floor learning rate after decay                         |
| `n_embd`                      | 768    | Token embedding dimension                               |
| `n_head`                      | 12     | Attention heads                                         |
| `n_layer`                     | 12     | Transformer blocks                                      |
| `dropout`                     | 0.0    | Regularization probability                              |
| `gradient_accumulation_steps` | 15     | Micro-batches per optimizer step (effective batch = 60) |
| `mixed_precision`             | fp16   | AMP autocast + GradScaler on CUDA                       |

<br>

## Dataset Notes
`prepare_data.py` downloads the full [OpenWebText](https://huggingface.co/datasets/openwebtext) corpus, tokenizes it with the GPT-2 BPE tokenizer from tiktoken, and writes the result as memory-mapped uint16 numpy arrays (`data/train.bin` and `data/val.bin`). The training script loads these files efficiently via `np.memmap` for random-access batching without loading the entire dataset into RAM.

`training_data.json` contains conversational examples as `{"user": "...", "assistant": "..."}` pairs used by `finetune.py` to adapt the pretrained model into a dedicated chatbot.

<br>

## KGPT-Lite Notebook

`kgpt-lite.ipynb` is a self-contained notebook that runs pretraining, fine-tuning, and inference end-to-end in a single Kaggle session. The model architecture and code are **identical** to the `.py` files — the only differences are training parameters tuned to fit within Kaggle's 10-hour T4 GPU limit:

| Parameter        | `.py` files | Notebook | Reason                                    |
| ---------------- | ----------- | -------- | ----------------------------------------- |
| `max_iters`      | 50,000      | 3,000    | Complete within a single 10-hour session  |
| `eval_interval`  | 2,000       | 500      | More frequent eval with fewer total iters |
| `warmup_iters`   | 2,000       | 200      | Proportional to shorter training run      |
| `lr_decay_iters` | 50,000      | 3,000    | Matches reduced `max_iters`               |

Fine-tuning and inference parameters also differ to improve chatbot quality on the smaller training budget:

| Parameter                | `.py` files | Notebook | Reason                                  |
| ------------------------ | ----------- | -------- | --------------------------------------- |
| `finetune_iters`         | 3,000       | 6,000    | More iterations for better convergence  |
| `finetune_lr`            | 1e-5        | 5e-5     | Higher LR so the model learns patterns  |
| `finetune_warmup`        | 100         | 200      | Proportional to longer fine-tuning run  |
| `inf_temperature`        | 0.7         | 0.3      | Lower randomness for coherent responses |
| `inf_top_k`              | 50          | 20       | Narrower sampling for 124M param model  |
| `inf_repetition_penalty` | 1.2         | 1.3      | Stronger dedup to prevent loops         |
| `inf_max_new_tokens`     | 256         | 128      | Single-sentence responses need fewer    |

Everything else — architecture, optimizer, batch size, gradient accumulation, mixed precision — is exactly the same.

**Additional notebook differences:**
- **Training data is generated at runtime.** The notebook embeds the full data generator inline (10,000 diverse Q&A pairs across 22 categories) instead of loading `training_data.json` from a file. This eliminates the need to upload the JSON to Kaggle.
- **Inference is single-sentence.** Responses are truncated to the first complete sentence for concise, practical chatbot output.
