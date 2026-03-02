![image](https://github.com/mytechnotalent/kgpt/blob/main/KGPT.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# KGPT
A GPT-2-class language model trained from scratch on [OpenWebText](https://huggingface.co/datasets/openwebtext) based on [Zero To Hero](https://karpathy.ai/zero-to-hero.html) utilizing tiktoken with the intent to augment AI Transformer-model education and reverse engineer GPT models from scratch.

The model matches the GPT-2 small architecture (`n_embd=768`, `n_head=12`, `n_layer=12`, `block_size=1024`, ~124 M parameters) and trains on the full OpenWebText dataset. After pretraining the model can be fine-tuned on conversational data to produce a real chatbot.

<br>

## Repository Files
| File                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `model.py`           | Shared GPT-2 architecture (Head, Block, BigramLanguageModel) |
| `train.py`           | Pretrains the model on OpenWebText                           |
| `finetune.py`        | Fine-tunes the pretrained model on training_data.json        |
| `inference.py`       | Professional-grade interactive chatbot                       |
| `prepare_data.py`    | Downloads OpenWebText and creates tokenized binary files     |
| `training_data.json` | Conversational dataset (user / assistant pairs)              |
| `pyproject.toml`     | Project metadata and dependencies                            |

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
2. Train the transformer for 600 000 iterations with learning-rate warmup and cosine decay, printing loss every 2 000 steps.
3. Save the pretrained weights to `kgpt_pretrained.pt`.

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
| Parameter                     | Value   | Purpose                            |
| ----------------------------- | ------- | ---------------------------------- |
| `batch_size`                  | 12      | Parallel sequences per micro-batch |
| `block_size`                  | 1024    | Maximum context length             |
| `max_iters`                   | 600 000 | Total training iterations          |
| `learning_rate`               | 6e-4    | Peak AdamW step size               |
| `warmup_iters`                | 2 000   | Linear LR warmup iterations        |
| `lr_decay_iters`              | 600 000 | Cosine decay horizon               |
| `min_lr`                      | 6e-5    | Floor learning rate after decay    |
| `n_embd`                      | 768     | Token embedding dimension          |
| `n_head`                      | 12      | Attention heads                    |
| `n_layer`                     | 12      | Transformer blocks                 |
| `dropout`                     | 0.0     | Regularization probability         |
| `gradient_accumulation_steps` | 5       | Micro-batches per optimizer step   |

<br>

## Dataset Notes
`prepare_data.py` downloads the full [OpenWebText](https://huggingface.co/datasets/openwebtext) corpus, tokenizes it with the GPT-2 BPE tokenizer from tiktoken, and writes the result as memory-mapped uint16 numpy arrays (`data/train.bin` and `data/val.bin`). The training script loads these files efficiently via `np.memmap` for random-access batching without loading the entire dataset into RAM.

`training_data.json` contains conversational examples as `{"user": "...", "assistant": "..."}` pairs used by `finetune.py` to adapt the pretrained model into a dedicated chatbot.

The legacy `data.txt` file contains conversational examples alternating `K:` and `Bot:` speaker tags.
