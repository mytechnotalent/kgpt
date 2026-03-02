import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# the device parameter specifies the device on which the model and tensors are placed for
# computation and if CUDA is available and enabled, the model will be placed on the GPU ('cuda'),
# if Apple MPS is available and enabled, the model will be placed on the Apple Silicon GPU ('mps'),
# which can significantly accelerate training and if neither is available or enabled,
# the model will be placed on the CPU ('cpu') and consider when choosing the appropriate device depends
# on the availability of compatible hardware and the memory requirements of the model
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# create a mapping from subwords to integers using the GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")

# the n_embd parameter represents the embedding dimension or size of the token embeddings
# in the model and it determines the dimensionality of the learned token representations and
# changing this parameter can affect the model's capacity to capture and encode information from
# the input tokens and a larger n_embd value allows for a higher capacity model but may increase the
# number of parameters and computational requirements, conversely, decreasing n_embd can result in a model
# with lower capacity and less expressive power
n_embd = 768
# the n_head parameter determines the number of attention heads used in the multi-head attention
# mechanism of the model and attention heads allow the model to attend to different parts of the input
# sequence simultaneously capturing different dependencies and patterns and increasing n_head allows
# for more fine-grained attention and enhances the model's ability to capture complex relationships,
# however, it also increases the computational cost and the number of parameters in the model
n_head = 12
# the n_layer parameter specifies the number of transformer blocks or layers in the model and
# each transformer block consists of attention mechanisms and feed-forward neural networks and
# increasing n_layer allows for a deeper model with more complex representations and increased
# modeling capacity, however, a higher number of layers may increase the computational requirements
# and the risk of overfitting if the model becomes too complex for the available training data
n_layer = 12
# the block_size parameter defines the maximum context length for predictions
# and it determines the number of tokens from the input sequence that the model
# considers when making predictions and if the context length exceeds the block_size,
# the model will only consider the most recent block_size tokens and
# when you change this parameter you can affect the model's ability to capture long-range
# dependencies in the input sequences and a larger block_size allows for more context but
# may also increase computational requirements
block_size = 1024
# the dropout parameter represents the dropout probability, which determines the probability
# of randomly setting inputs to zero during training and dropout is a regularization technique
# that helps prevent overfitting by reducing co-adaptation between neurons and a dropout value
# of 0.0 means no dropout is applied, while a value of 1.0 means all inputs are set to zero and
# adjusting the dropout value can influence the model's generalization ability and higher dropout
# values introduce more regularization, which can be useful when dealing with limited training
# data or to prevent overfitting, however, too much dropout may lead to underfitting, and too
# little dropout may result in overfitting and GPT-2 was originally trained with dropout of 0.0
# because the OpenWebText dataset is large enough to not require dropout regularization
dropout = 0.0


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
        """
        Initializes a single head of self-attention.

        Args:
            head_size (int): The size of the attention head determining the
                dimensionality of the key, query, and value projections.
        """
        super().__init__()
        # linear layers for key, query, and value projections
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # lower triangular mask for masking attention scores
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def _compute_attention_weights(self, q, k, T, C):
        """
        Computes masked and normalized attention weight matrix from queries and keys.

        Args:
            q (torch.Tensor): Query tensor of shape (B, T, C).
            k (torch.Tensor): Key tensor of shape (B, T, C).
            T (int): Current sequence length.
            C (int): Head size dimensionality for scaling.

        Returns:
            torch.Tensor: Attention weights of shape (B, T, T) after masking,
                softmax normalization, and dropout.
        """
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        return self.dropout(wei)

    def forward(self, x):
        """
        Performs the forward pass of the attention head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length,
                embedding_size).

        Returns:
            torch.Tensor: Output tensor after attention computation of shape
                (batch_size, sequence_length, head_size).
        """
        _, T, C = x.shape
        # compute key, query, and value projections
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        wei = self._compute_attention_weights(q, k, T, C)
        # perform weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        return wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)


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
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length,
                embedding_size).

        Returns:
            torch.Tensor: Output tensor after the multi-head attention computation
                of shape (batch_size, sequence_length, embedding_size).
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
        net (nn.Sequential): Sequential module containing linear layers, GELU activation,
            and dropout.

    Methods:
        forward(x): Performs the forward pass of the feed-forward module.

    Example:
        # Create a feed-forward module
        ff_module = FeedForward(n_embd=768)

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
        # sequential module containing linear layers, GELU activation, and dropout
        # using GELU instead of ReLU to match the GPT-2 architecture
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Performs the forward pass of the feed-forward module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length,
                embedding_size).

        Returns:
            torch.Tensor: Output tensor after the feed-forward computation of shape
                (batch_size, sequence_length, embedding_size).
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
        block = Block(n_embd=768, n_head=12)

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
        # multi-head self-attention module
        self.sa = MultiHeadAttention(n_head, head_size)
        # feed-forward module
        self.ffwd = FeedForward(n_embd)
        # layer normalization modules
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Performs the forward pass of the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length,
                embedding_size).

        Returns:
            torch.Tensor: Output tensor after the transformer block computation
                of shape (batch_size, sequence_length, embedding_size).
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    """
    A GPT-2-class language model based on the Transformer architecture trained
    from scratch on OpenWebText data.

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
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, enc.n_vocab)

    def _compute_logits(self, idx):
        """
        Computes the output logits from input token indices through the full
        transformer stack including embeddings, blocks, and the language model head.

        Args:
            idx (torch.Tensor): Input indices tensor of shape (B, T).

        Returns:
            torch.Tensor: Output logits tensor of shape (B, T, vocab_size).
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def _compute_loss(self, logits, targets):
        """
        Computes the cross-entropy loss between predicted logits and target indices
        by reshaping the tensors into a two-dimensional classification format.

        Args:
            logits (torch.Tensor): Predicted logits of shape (B, T, vocab_size).
            targets (torch.Tensor): Target indices of shape (B, T).

        Returns:
            torch.Tensor: Scalar cross-entropy loss value.
        """
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        return F.cross_entropy(logits, targets)

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
        logits = self._compute_logits(idx)
        loss = self._compute_loss(logits, targets) if targets is not None else None
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
