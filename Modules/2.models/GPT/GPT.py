import torch
import torch.nn as nn
from torch.nn import functional as F

# from ..configs import GPT_CONFIG_124M

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism. This module splits the input into multiple heads,
    performs scaled dot-product attention for each head, and combines the results.

    Parameters:
    - d_in: int, the dimensionality of the input
    - d_out: int, the dimensionality of the output (must be divisible by num_heads)
    - context_length: int, the length of the context window for causal masking
    - dropout: float, dropout probability
    - num_heads: int, number of attention heads
    - qkv_bias: bool, whether to include bias in query, key, and value projections

    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # Attention head configuration
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Each head has a reduced dimensionality

        # Linear projections for queries, keys, and values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection after concatenating heads
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Causal mask to prevent looking ahead (upper triangular matrix)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        Forward pass through multi-head attention.
        
        Args:
        - x: Tensor of shape (batch_size, num_tokens, d_in), input to the attention mechanism

        Returns:
        - context_vec: Tensor of shape (batch_size, num_tokens, d_out), output of the attention
        """
        b, num_tokens, d_in = x.shape

        # Compute keys, queries, and values
        keys = self.W_key(x)      # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split into multiple heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose for attention computation
        keys = keys.transpose(1, 2)        # Shape: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)  # Shape: (b, num_heads, num_tokens, num_tokens)
        
        # Apply the causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Apply softmax and compute attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute the context vector
        context_vec = (attn_weights @ values).transpose(1, 2)  # Shape: (b, num_tokens, num_heads, head_dim)

        # Combine heads by flattening them
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Final linear projection
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    """
    Layer normalization, a technique to normalize the activations for each input example.

    Parameters:
    - emb_dim: int, the dimensionality of the input embeddings
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # Small value to avoid division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))  # Scaling parameter
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # Shifting parameter

    def forward(self, x):
        """
        Forward pass through layer normalization.
        
        Args:
        - x: Tensor of shape (batch_size, num_tokens, emb_dim), input to be normalized

        Returns:
        - Normalized tensor with the same shape as input
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass through GELU activation.
        
        Args:
        - x: Input tensor
        
        Returns:
        - Tensor after applying GELU
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Feed-forward network consisting of two linear layers with a GELU activation in between.

    Parameters:
    - cfg: dict, configuration dictionary containing 'emb_dim' for the dimensionality of the layers
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        Forward pass through the feed-forward network.
        
        Args:
        - x: Input tensor
        
        Returns:
        - Output tensor after passing through the linear layers
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Single block of a Transformer model. It contains a multi-head attention layer, 
    a feed-forward network, and layer normalization with residual connections.

    Parameters:
    - cfg: dict, configuration dictionary containing:
      'emb_dim', 'context_length', 'n_heads', 'drop_rate', and 'qkv_bias'
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        Forward pass through the Transformer block.
        
        Args:
        - x: Tensor of shape (batch_size, num_tokens, emb_dim), input to the block

        Returns:
        - Tensor after passing through attention and feed-forward layers with residual connections
        """
        # Attention block with shortcut
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Attention
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        # Feed-forward block with shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        return x


class GPTModel(nn.Module):
    """
    GPT-style language model consisting of embedding layers, multiple Transformer blocks, 
    and a final linear projection to produce logits over the vocabulary.

    Parameters:
    - cfg: dict, configuration dictionary containing:
      'vocab_size', 'emb_dim', 'context_length', 'n_layers', and 'drop_rate'
    """
    def __init__(self, cfg):
        super().__init__()
        # Embedding layers for tokens and positions
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final layer normalization and linear output layer
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        Forward pass through the GPT model.
        
        Args:
        - in_idx: Tensor of shape (batch_size, seq_len), input sequence of token indices

        Returns:
        - logits: Tensor of shape (batch_size, seq_len, vocab_size), output logits over the vocabulary
        """
        b, seq_len = in_idx.shape
        assert seq_len <= self.cfg["context_length"], "Sequence length exceeds context length."

        # Compute token and position embeddings
        tok_emb = self.tok_emb(in_idx)  # Token embeddings: (b, seq_len, emb_dim)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # Position embeddings

        # Combine embeddings and apply dropout
        emb = self.drop_emb(tok_emb + pos_emb)

        # Pass through the Transformer blocks
        x = self.trf_blocks(emb)

        # Apply final layer normalization and compute logits
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


