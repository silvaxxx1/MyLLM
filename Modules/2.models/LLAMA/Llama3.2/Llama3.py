# Llama3.py

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom modules
from Llama3_utils import SharedBuffers , apply_rotary_embeddings


# FeedForward Class
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize the first linear layer: projects input embedding to hidden dimension
        self.fc1 = nn.Linear(config['emb_dim'], config['hidden_dim'], bias=False)
        
        # Initialize the second linear layer: creates additional projections for gating mechanism
        self.fc2 = nn.Linear(config['emb_dim'], config['hidden_dim'], bias=False)
        
        # Initialize the third linear layer: maps back from hidden dimension to original embedding dimension
        self.fc3 = nn.Linear(config['hidden_dim'], config['emb_dim'], bias=False)

    def forward(self, x):
        # x: (b, num_tokens, emb_dim)
        x1 = self.fc1(x)           # (b, num_tokens, hidden_dim)
        x2 = self.fc2(x)           # (b, num_tokens, hidden_dim)
        # SiLU gate: element-wise activation then multiply
        x = F.silu(x1) * x2        # (b, num_tokens, hidden_dim)
        x = self.fc3(x)            # (b, num_tokens, emb_dim)
        return x


# GroupedQueryAttention Class
class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, context_length, num_heads,
            num_kv_groups, rope_base=10_000, rope_config=None,
            dtype=None, device='cpu'  # Device is specified with a default value
        ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        # Initialize essential parameters
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.device = device

        # Define linear layers for keys, values, and queries
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        
        # Final projection layer after attention
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        # Configure the grouping structure for keys and values
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        # Shared buffers for causal mask and rotary embeddings
        mask, freqs_complex = SharedBuffers.get_buffers(
            context_length, self.head_dim, rope_base, rope_config, dtype=dtype, device=device
        )
        self.register_buffer("mask", mask)
        self.register_buffer("freqs_complex", freqs_complex)

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # x: (b, num_tokens, d_in)

        # Compute queries, keys, and values
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        keys    = self.W_key(x)    # (b, num_tokens, num_kv_groups * head_dim)
        values  = self.W_value(x)  # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape to support multi-head processing
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)     # (b, num_tokens, num_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)    # (b, num_tokens, num_kv_groups, head_dim)
        values  = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)  # (b, num_tokens, num_kv_groups, head_dim)

        # Transpose for attention computation
        keys    = keys.transpose(1, 2)    # (b, num_kv_groups, num_tokens, head_dim)
        values  = values.transpose(1, 2)  # (b, num_kv_groups, num_tokens, head_dim)
        queries = queries.transpose(1, 2) # (b, num_heads, num_tokens, head_dim)

        # Apply rotary embeddings for positional information
        keys    = apply_rotary_embeddings(keys, self.freqs_complex, self.device)    # (b, num_kv_groups, num_tokens, head_dim)
        queries = apply_rotary_embeddings(queries, self.freqs_complex, self.device) # (b, num_heads, num_tokens, head_dim)

        # Expand keys and values from num_kv_groups to num_heads (GQA broadcast)
        keys   = keys.repeat_interleave(self.group_size, dim=1)    # (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # (b, num_heads, num_tokens, head_dim)

        # Compute attention scores and apply causal mask
        attn_scores = queries @ keys.transpose(2, 3)                 # (b, num_heads, num_tokens, num_tokens)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]       # (num_tokens, num_tokens)
        attn_scores.masked_fill_(mask_bool, -torch.inf)              # (b, num_heads, num_tokens, num_tokens)

        # Normalize attention scores and compute weighted sum of values
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)  # (b, num_heads, num_tokens, num_tokens)
        context_vec = (attn_weights @ values).transpose(1, 2)                    # (b, num_tokens, num_heads, head_dim)

        # Combine heads and apply final projection
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)  # (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec)                       # (b, num_tokens, d_out)
        return context_vec


# TransformerBlock Class
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize multi-head attention layer
        self.att = GroupedQueryAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            num_heads=config["n_heads"],
            num_kv_groups=config["n_kv_groups"],
            rope_base=config["rope_base"],
            rope_config=config["rope_freq"],
            dtype=config["dtype"]
        )
        
        # Initialize feedforward network
        self.ff = FeedForward(config)
        
        # Layer normalization layers
        self.norm1 = nn.RMSNorm(config['emb_dim'])
        self.norm2 = nn.RMSNorm(config['emb_dim'])

    def forward(self, x):
        # x: (b, num_tokens, emb_dim)
        shortcut = x                               # (b, num_tokens, emb_dim)
        x = self.norm1(x)                          # (b, num_tokens, emb_dim)
        x = self.att(x.to(torch.bfloat16))        # (b, num_tokens, emb_dim)
        x = x + shortcut                           # (b, num_tokens, emb_dim)

        shortcut = x                               # (b, num_tokens, emb_dim)
        x = self.norm2(x)                          # (b, num_tokens, emb_dim)
        x = self.ff(x.to(torch.bfloat16))         # (b, num_tokens, emb_dim)
        x = x + shortcut                           # (b, num_tokens, emb_dim)
        return x


# Llama3 Class
class Llama3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding layer to transform input tokens to dense vectors
        self.token_embedding = nn.Embedding(
            config['vocab_size'], config['emb_dim'], dtype=config['dtype']
        )
        
        # Stack of transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )
        
        # Final layer normalization
        self.final_norm = nn.RMSNorm(config['emb_dim'])
        
        # Output projection to vocabulary size
        self.out_head = nn.Linear(
            config['emb_dim'], config['vocab_size'], bias=False, dtype=config['dtype']
        )

    def forward(self, x):
        # x: (b, num_tokens)
        tok_emb = self.token_embedding(x)   # (b, num_tokens, emb_dim)

        # Pass through transformer blocks: each preserves (b, num_tokens, emb_dim)
        x = self.trf_blocks(tok_emb)        # (b, num_tokens, emb_dim)

        x      = self.final_norm(x)                  # (b, num_tokens, emb_dim)
        logits = self.out_head(x.to(torch.bfloat16)) # (b, num_tokens, vocab_size)
        return logits
