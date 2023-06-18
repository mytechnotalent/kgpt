![image](https://github.com/mytechnotalent/kgpt/blob/main/KGPT.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>
# KGPT
A custom GPT based on [Zero To Hero](https://karpathy.ai/zero-to-hero.html) utilizing Tiktoken with the intent to augment AI Transformer-model education and reverse engineer GPT models from scratch.

## setup venv
```
python -m venv venv
```

## visit pytorch.org and select your system config
```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

## install tiktoken
```
pip install tiktoken
```

## `kgpt.py`
```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import warnings

# ignore cuda warnings
warnings.filterwarnings('ignore')

# the batch_size parameter determines how many independent sequences will be 
# processed in parallel during training and increasing the batch size allows 
# for more efficient computation and parallelization but may require more memory 
# and a larger batch size can also provide a more stable gradient estimation 
# but might lead to slower convergence or generalization issues
batch_size = 8  # how many independent sequences will we process in parallel?
# the block_size parameter defines the maximum context length for predictions
# and tt determines the number of tokens from the input sequence that the model 
# considers when making predictions and if the context length exceeds the block_size, 
# the model will only consider the most recent block_size tokens
# when you change this parameter you can affect the model's ability to capture long-range 
# dependencies in the input sequences and a larger block_size allows for more context but 
# may also increase computational requirements
block_size = 64  # what is the maximum context length for predictions?
# the max_iters parameter represents the maximum number of iterations or steps during the 
# training process and tt determines how many times the model will update its parameters 
# based on the training data and increasing max_iters allows for more training iterations, 
# potentially leading to better model performance, however, it may also increase the training 
# time and the risk of overfitting if the model starts memorizing the training data
max_iters = 500
# the eval_interval parameter specifies the frequency at which the model's performance 
# is evaluated on the training and validation sets and it determines how often the loss 
# values are printed or logged during training and a smaller eval_interval provides more 
# frequent updates on the model's progress but can increase the computational overhead
# and adjusting this parameter depends on the desired level of monitoring and the trade-off 
# between evaluation frequency and training efficiency
eval_interval = 100
# the learning_rate parameter controls the step size at each iteration during the model's 
# parameter update using gradient descent optimization and it determines how much the model's 
# parameters are adjusted based on the computed gradients and a higher learning rate allows 
# for larger updates, potentially leading to faster convergence, however, using a very high 
# learning rate can cause the optimization process to become unstable or prevent 
# convergence. on the other hand, a lower learning rate may require more iterations for 
# convergence but can provide more precise parameter updates
learning_rate = 1e-3
# the device parameter specifies the device on which the model and tensors are placed for 
# computation and if CUDA is available and enabled, the model will be placed on the GPU ('cuda'), 
# which can significantly accelerate training and if CUDA is not available or enabled, 
# the model will be placed on the CPU ('cpu') when choosing the appropriate device depends 
# on the availability of compatible hardware and the memory requirements of the model
device = 'cuda' if torch.backends.cuda.is_built() else 'cpu'
# the eval_iters parameter determines the number of iterations used to estimate the loss on the 
# training and validation sets during evaluation and It represents the number of iterations used 
# to compute the average loss value and a larger eval_iters value provides a more accurate estimation 
# of the loss but can increase the evaluation time and adjusting this parameter depends on the 
# desired level of accuracy in the loss estimation and the trade-off between evaluation time and accuracy
eval_iters = 200
# the n_embd parameter represents the embedding dimension or size of the token embeddings 
# in the model and it determines the dimensionality of the learned token representations and 
# changing this parameter can affect the model's capacity to capture and encode information from 
# the input tokens and a larger n_embd value allows for a higher capacity model but may increase the 
# number of parameters and computational requirements, conversely, decreasing n_embd can result in a model 
# with lower capacity and less expressive power
n_embd = 64
# the n_head parameter determines the number of attention heads used in the multi-head attention
# mechanism of the model and attention heads allow the model to attend to different parts of the input 
# sequence simultaneously, capturing different dependencies and patterns. Increasing n_head allows 
# for more fine-grained attention and enhances the model's ability to capture complex relationships, 
# however, it also increases the computational cost and the number of parameters in the model
n_head = 4
# the n_layer parameter specifies the number of transformer blocks or layers in the model and 
# each transformer block consists of attention mechanisms and feed-forward neural networks and 
# increasing n_layer allows for a deeper model with more complex representations and increased 
# modeling capacity, however, a higher number of layers may increase the computational requirements 
# and the risk of overfitting if the model becomes too complex for the available training data
n_layer = 4
# the dropout parameter represents the dropout probability, which determines the probability 
# of randomly setting inputs to zero during training and dropout is a regularization technique 
# that helps prevent overfitting by reducing co-adaptation between neurons and a dropout value 
# of 0.0 means no dropout is applied, while a value of 1.0 means all inputs are set to zero and 
# adjusting the dropout value can influence the model's generalization ability and higher dropout 
# values introduce more regularization, which can be useful when dealing with limited training
# data or to prevent overfitting, however, too much dropout may lead to underfitting, and too 
# little dropout may result in overfitting
dropout = 0.0

# print device either cuda or cpu
print(device)

# torch.manual_seed(1337)  # if you want to have reproducibility

# open dataset and create text object
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# create a mapping from subwords to integers
enc = tiktoken.get_encoding("gpt2")

# train and test splits
data = torch.tensor(enc.encode(text), dtype=torch.long)
n = int(0.8*len(data))  # first 80% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """
    Retrieves a batch of data for a given split.

    Args:
        split (str): Specifies the split of the data to retrieve ('train' or 'val').

    Returns:
        tuple: A tuple containing the input data and corresponding target data.
            - x (torch.Tensor): Input data of shape (batch_size, block_size).
            - y (torch.Tensor): Target data of shape (batch_size, block_size).

    Raises:
        ValueError: If an invalid split value is provided.

    Notes:
        - The function assumes the existence of the variables `train_data`, `val_data`,
          `block_size`, `batch_size`, and `device` in the global scope.
        - `train_data` and `val_data` are expected to be PyTorch tensors containing the
          complete training and validation datasets, respectively.
        - `block_size` specifies the length of each sequence block in the data.
        - `batch_size` determines the number of sequences to include in each batch.
        - `device` specifies the device on which the tensors will be placed.

    Example:
        # Retrieve a training batch
        x_train, y_train = get_batch('train')

        # Retrieve a validation batch
        x_val, y_val = get_batch('val')
    """
    data = train_data if split == 'train' else val_data
    # randomly select starting indices for the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # retrieve the input and target sequences for each starting index
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # move the tensors to the specified device
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """
    Estimates the average loss on the training and validation datasets.

    Returns:
        dict: A dictionary containing the average loss for each dataset split.
            - 'train' (float): Average loss on the training dataset.
            - 'val' (float): Average loss on the validation dataset.

    Notes:
        - The function assumes the existence of the variables `eval_iters` and `model`
          in the global scope.
        - `eval_iters` specifies the number of iterations to perform for loss estimation.
        - `model` is the PyTorch model object to evaluate.
        - The `get_batch` function is expected to be defined and accessible.

    Example:
        # Estimate the losses
        loss_estimation = estimate_loss()

        # Access the average loss on the training dataset
        train_loss = loss_estimation['train']

        # Access the average loss on the validation dataset
        val_loss = loss_estimation['val']
    """
    out = {}
    model.eval()
    # estimate losses for each dataset split
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        # calculate the average loss
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """
    A single head of self-attention.

    Args:
        head_size (int): The size of the attention head.

    Attributes:
        key (nn.Linear): Linear layer for computing the 'key' projection.
        query (nn.Linear): Linear layer for computing the 'query' projection.
        value (nn.Linear): Linear layer for computing the 'value' projection.
        tril (torch.Tensor): Lower triangular mask for masking attention scores.
        dropout (nn.Dropout): Dropout layer for regularization.

    Methods:
        forward(x): Performs the forward pass of the attention head.

    Example:
        # Create an attention head
        head = Head(head_size=128)

        # Perform the forward pass
        output = head(x)
    """

    def __init__(self, head_size):
        super().__init__()

        # Linear layers for key, query, and value projections
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Lower triangular mask for masking attention scores
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Performs the forward pass of the attention head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).

        Returns:
            torch.Tensor: Output tensor after attention computation of shape (batch_size, sequence_length, embedding_size).
        """
        _, T, C = x.shape
        # compute key, query, and value projections
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.

    Args:
        num_heads (int): The number of attention heads.
        head_size (int): The size of each attention head.

    Attributes:
        heads (nn.ModuleList): List of attention heads.
        proj (nn.Linear): Linear layer for projecting the concatenated heads.
        dropout (nn.Dropout): Dropout layer for regularization.

    Methods:
        forward(x): Performs the forward pass of the multi-head attention module.

    Example:
        # Create a multi-head attention module
        attention = MultiHeadAttention(num_heads=8, head_size=64)

        # Perform the forward pass
        output = attention(x)
    """

    def __init__(self, num_heads, head_size):
        """
        Initializes a multi-head attention module.

        Args:
            num_heads (int): The number of attention heads.
            head_size (int): The size of each attention head.
        """
        super().__init__()
        # list of attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # linear layer for projecting the concatenated heads
        self.proj = nn.Linear(n_embd, n_embd)
        # dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Performs the forward pass of the multi-head attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).

        Returns:
            torch.Tensor: Output tensor after the multi-head attention computation of shape (batch_size, sequence_length, embedding_size).
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Feed-forward module consisting of linear layers followed by a non-linearity and dropout.

    Args:
        n_embd (int): The input and output embedding size.

    Attributes:
        net (nn.Sequential): Sequential module containing linear layers, ReLU activation, and dropout.

    Methods:
        forward(x): Performs the forward pass of the feed-forward module.

    Example:
        # Create a feed-forward module
        ff_module = FeedForward(n_embd=512)

        # Perform the forward pass
        output = ff_module(x)
    """

    def __init__(self, n_embd):
        """
        Initializes a feed-forward module.

        Args:
            n_embd (int): The input and output embedding size.
        """
        super().__init__()
        # sequential module containing linear layers, ReLU activation, and dropout
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Performs the forward pass of the feed-forward module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).

        Returns:
            torch.Tensor: Output tensor after the feed-forward computation of shape (batch_size, sequence_length, embedding_size).
        """
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block consisting of self-attention and feed-forward layers.

    Args:
        n_embd (int): The embedding dimension.
        n_head (int): The number of attention heads.

    Attributes:
        sa (MultiHeadAttention): Multi-head self-attention module.
        ffwd (FeedForward): Feed-forward module.
        ln1 (nn.LayerNorm): Layer normalization module after the self-attention layer.
        ln2 (nn.LayerNorm): Layer normalization module after the feed-forward layer.

    Methods:
        forward(x): Performs the forward pass of the transformer block.

    Example:
        # Create a transformer block
        block = Block(n_embd=512, n_head=8)

        # Perform the forward pass
        output = block(x)
    """

    def __init__(self, n_embd, n_head):
        """
        Initializes a Transformer block.

        Args:
            n_embd (int): The embedding dimension.
            n_head (int): The number of attention heads.
        """
        super().__init__()

        head_size = n_embd // n_head

        # Multi-head self-attention module
        self.sa = MultiHeadAttention(n_head, head_size)

        # Feed-forward module
        self.ffwd = FeedForward(n_embd)

        # Layer normalization modules
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Performs the forward pass of the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).

        Returns:
            torch.Tensor: Output tensor after the transformer block computation of shape (batch_size, sequence_length, embedding_size).
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    """
    A simple bigram language model based on the Transformer architecture.

    Args:
        None

    Attributes:
        token_embedding_table (nn.Embedding): Lookup table for token embeddings.
        position_embedding_table (nn.Embedding): Lookup table for position embeddings.
        blocks (nn.Sequential): Sequence of Transformer blocks.
        ln_f (nn.LayerNorm): Final layer normalization.
        lm_head (nn.Linear): Linear layer for language model prediction.

    Methods:
        __init__():
            Initializes the BigramLanguageModel class.

        forward(idx, targets=None):
            Performs forward pass through the model.

        generate(idx, max_new_tokens):
            Generates new tokens based on the given context.

    """

    def __init__(self):
        """
        Initializes the BigramLanguageModel class by setting up the model architecture.

        Args:
            None

        Returns:
            None
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(enc.n_vocab, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, enc.n_vocab)

    def forward(self, idx, targets=None):
        """
        Performs forward pass through the model.

        Args:
            idx (torch.Tensor): Input indices tensor of shape (B, T).
            targets (torch.Tensor): Target indices tensor of shape (B, T).

        Returns:
            logits (torch.Tensor): Output logits tensor of shape (B, T, vocab_size).
            loss (torch.Tensor or None): Optional loss tensor if targets are provided.
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generates new tokens based on the given context.

        Args:
            idx (torch.Tensor): Input indices tensor of shape (B, T).
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            idx (torch.Tensor): Generated indices tensor of shape (B, T+max_new_tokens).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# create instance of the BigramLanguageModel class and assign it to the variable model with default settings
model = BigramLanguageModel()

# move the model to the specified device to ensure that the model and its parameters are stored and 
# operated on using the appropriate hardware (e.g., GPU if available) and the modified model is assigned 
# to the variable m.
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# loop iterates over a specified number of iterations (max_iters)
# used for training the model and performing updates on the parameters
for iter in range(max_iters):
    # checks if it's time to evaluate the loss on the training and 
    # validation sets and tt is determined by the value of eval_interval 
    # or if it's the last iteration (iter == max_iters - 1)
    # the estimate_loss() function is called to compute the losses, and 
    # then the losses are printed to provide feedback on the model's performance
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # sample a batch of data (xb) and its corresponding targets (yb) 
    # from the training set using the get_batch() function
    # the returned tensors represent inputs and targets for the model
    xb, yb = get_batch('train')
    # evaluate the loss with the batch of inputs and targets (xb, yb) is passed 
    # to the model (model) to obtain the predicted logits and the computed loss 
    # and the logits represent the model's output probabilities for the next token 
    # prediction
    logits, loss = model(xb, yb)
    # set all the gradients of the optimizer's parameters to zero and
    # prepare the optimizer for the next iteration to avoid accumulating gradients 
    # from previous iterations
    optimizer.zero_grad(set_to_none=True)
    # compute the gradients of the loss with respect to the model's parameters using
    # backpropagation and the gradients are used to update the parameters during the 
    # optimizer's step() operation
    loss.backward()
    # update the model's parameters based on the computed gradients and the optimization 
    # algorithm implemented by the optimizer. It performs a step of gradient descent 
    # to minimize the loss and improve the model's performance.
    optimizer.step()

# initialize a tensor context with shape (1, 1) filled with zeros and the tensor is 
# of type torch.long (representing integer values) and is placed on the specified 
# device (such as CPU or GPU) then this tensor is used as the initial context for 
# generating new tokens
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# generate a sequence of tokens using the generate method of the BigramLanguageModel 
# instance (m)and the method takes the context tensor and a maximum number of new tokens 
# to generate (max_new_tokens=2000) and the generated sequence is obtained as a tensor of shape 
# (1, T+1) where T is the number of tokens in the generated sequence and the .tolist() method
# converts the tensor to a Python list and the enc.decode() function is then used to decode the 
# list of tokens back into human-readable text then it maps the token indices to their 
# corresponding string representations based on the encoding used by the model and finally
# the resulting decoded text is printed to the console, displaying the generated sequence of 
# tokens as text
print(enc.decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
```

## License
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
