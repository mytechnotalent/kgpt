import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# create a mapping from subwords to integers using the GPT-2 BPE tokenizer which
# provides the vocabulary size needed for the embedding and output layers
enc = tiktoken.get_encoding("gpt2")

# the n_embd parameter represents the embedding dimension determining the
# dimensionality of learned token representations throughout the model
n_embd = 768
# the n_head parameter determines the number of attention heads in the multi-head
# attention mechanism allowing the model to attend to different patterns simultaneously
n_head = 12
# the n_layer parameter specifies the number of transformer blocks providing the
# depth of the model and its capacity for learning complex representations
n_layer = 12
# the block_size parameter defines the maximum context length for predictions
# determining how many tokens the model considers when generating each output
block_size = 1024
# the dropout parameter controls regularization during training where 0.0 means no
# dropout is applied matching the original GPT-2 pretraining configuration
dropout = 0.0


class CausalSelfAttention(nn.Module):
    """
    Fused multi-head causal self-attention module using flash attention for memory
    efficiency, combining all key, query, and value projections into a single
    linear layer matching the nanoGPT architecture.

    Args:
        n_embd (int): The embedding dimension size.
        n_head (int): The number of attention heads.

    Attributes:
        n_head (int): Number of attention heads stored for tensor reshaping.
        c_attn (nn.Linear): Fused linear projection for query, key, and value.
        c_proj (nn.Linear): Output projection with scaled initialization flag.
        attn_dropout (nn.Dropout): Dropout applied to attention weights.
        resid_dropout (nn.Dropout): Dropout applied to the output projection.

    Methods:
        forward(x): Performs the forward pass of the attention module.

    Example:
        attn = CausalSelfAttention(n_embd=768, n_head=12)
        output = attn(x)
    """

    def __init__(self, n_embd, n_head):
        """
        Initializes the fused causal self-attention module with a single linear
        layer for all query, key, and value projections.

        Args:
            n_embd (int): The embedding dimension size.
            n_head (int): The number of attention heads.
        """
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _compute_qkv(self, x):
        """
        Computes query, key, and value tensors from a single fused linear
        projection and reshapes them into multi-head format for attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            tuple: A tuple of (q, k, v) tensors each of shape
                (B, n_head, T, head_size).
        """
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        return q, k, v

    def forward(self, x):
        """
        Performs the forward pass computing causal self-attention using flash
        attention for memory-efficient scaled dot-product computation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after attention of shape (B, T, C).
        """
        B, T, C = x.shape
        q, k, v = self._compute_qkv(x)
        dp = self.attn_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dp)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))


class FeedForward(nn.Module):
    """
    Feed-forward module with GELU activation matching the GPT-2 architecture
    using c_fc and c_proj naming for nanoGPT compatibility.

    Args:
        n_embd (int): The input and output embedding size.

    Attributes:
        c_fc (nn.Linear): First linear projection expanding to 4x embedding size.
        gelu (nn.GELU): GELU activation function matching GPT-2.
        c_proj (nn.Linear): Output projection with scaled initialization flag.
        dropout (nn.Dropout): Dropout layer for regularization.

    Methods:
        forward(x): Performs the forward pass of the feed-forward module.

    Example:
        ff = FeedForward(n_embd=768)
        output = ff(x)
    """

    def __init__(self, n_embd):
        """
        Initializes the feed-forward module with GELU activation.

        Args:
            n_embd (int): The input and output embedding size.
        """
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Performs the forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """
    Transformer block with pre-norm architecture consisting of causal
    self-attention and feed-forward layers with residual connections.

    Args:
        n_embd (int): The embedding dimension.
        n_head (int): The number of attention heads.

    Attributes:
        ln1 (nn.LayerNorm): Layer normalization before self-attention.
        attn (CausalSelfAttention): Fused multi-head causal self-attention.
        ln2 (nn.LayerNorm): Layer normalization before feed-forward.
        mlp (FeedForward): Feed-forward module with GELU activation.

    Methods:
        forward(x): Performs the forward pass of the transformer block.

    Example:
        block = Block(n_embd=768, n_head=12)
        output = block(x)
    """

    def __init__(self, n_embd, n_head):
        """
        Initializes a transformer block with attention and feed-forward layers.

        Args:
            n_embd (int): The embedding dimension.
            n_head (int): The number of attention heads.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd)

    def forward(self, x):
        """
        Performs the forward pass with residual connections around each sublayer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    A GPT-2-class language model matching the nanoGPT architecture with weight
    tying between token embeddings and the language model head, custom scaled
    initialization for residual projections, and approximately 124M parameters.

    Attributes:
        token_embedding_table (nn.Embedding): Lookup table for token embeddings.
        position_embedding_table (nn.Embedding): Lookup table for position embeddings.
        blocks (nn.Sequential): Sequence of Transformer blocks.
        ln_f (nn.LayerNorm): Final layer normalization.
        lm_head (nn.Linear): Language model head sharing weights with token embeddings.

    Methods:
        forward(idx, targets=None): Performs forward pass through the model.
        generate(idx, max_new_tokens): Generates new tokens autoregressively.

    Example:
        model = GPT()
        logits, loss = model(idx, targets)
    """

    def _build_layers(self):
        """
        Creates all model layers including embeddings, transformer blocks,
        final layer norm, and the weight-tied language model head.

        Returns:
            None: Sets model layers as instance attributes.
        """
        self.token_embedding_table = nn.Embedding(enc.n_vocab, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, enc.n_vocab, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_linear(self, module):
        """
        Initializes a linear layer with normal distribution using standard
        deviation 0.02, scaled down for residual projections to prevent
        signal growth through the residual stream.

        Args:
            module (nn.Linear): The linear module to initialize.

        Returns:
            None: Modifies module weights in place.
        """
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * n_layer) ** -0.5
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def _init_embedding(self, module):
        """
        Initializes an embedding layer with normal distribution using standard
        deviation 0.02 matching the GPT-2 initialization scheme.

        Args:
            module (nn.Embedding): The embedding module to initialize.

        Returns:
            None: Modifies module weights in place.
        """
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_weights(self, module):
        """
        Dispatches weight initialization to the appropriate handler based on
        module type following the GPT-2 paper initialization scheme.

        Args:
            module (nn.Module): The module whose weights are being initialized.

        Returns:
            None: Delegates to type-specific initialization methods.
        """
        if isinstance(module, nn.Linear):
            self._init_linear(module)
        elif isinstance(module, nn.Embedding):
            self._init_embedding(module)

    def __init__(self):
        """
        Initializes the GPT model by building all layers and applying custom
        weight initialization with scaled residual projections.
        """
        super().__init__()
        self._build_layers()
        self.apply(self._init_weights)

    def _compute_logits(self, idx):
        """
        Computes output logits from input token indices through the full
        transformer stack including embeddings, blocks, and the language model head.

        Args:
            idx (torch.Tensor): Input indices tensor of shape (B, T).

        Returns:
            torch.Tensor: Output logits tensor of shape (B, T, vocab_size).
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def _compute_loss(self, logits, targets):
        """
        Computes cross-entropy loss between predicted logits and target indices
        by reshaping tensors into two-dimensional classification format.

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
        Performs forward pass computing logits and optionally cross-entropy loss.

        Args:
            idx (torch.Tensor): Input indices tensor of shape (B, T).
            targets (torch.Tensor): Optional target indices of shape (B, T).

        Returns:
            tuple: A tuple of (logits, loss) where logits has shape
                (B, T, vocab_size) and loss is a scalar tensor or None.
        """
        logits = self._compute_logits(idx)
        loss = self._compute_loss(logits, targets) if targets is not None else None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generates new tokens autoregressively based on the given context
        using multinomial sampling from the softmax distribution.

        Args:
            idx (torch.Tensor): Input indices tensor of shape (B, T).
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            torch.Tensor: Extended indices tensor of shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
