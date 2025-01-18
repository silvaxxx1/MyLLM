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
        # Compute first projection
        x1 = self.fc1(x)
        # Compute second projection for gating
        x2 = self.fc2(x)
        # Element-wise multiplication after SiLU activation introduces non-linear interactions
        x = F.silu(x1) * x2 
        # Project back to the original embedding space
        x = self.fc3(x)
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
        b, num_tokens, d_in = x.shape

        # Compute queries, keys, and values
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape to support multi-head processing
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose for compatibility with attention computation
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Apply rotary embeddings for positional information
        keys = apply_rotary_embeddings(keys, self.freqs_complex, self.device)
        queries = apply_rotary_embeddings(queries, self.freqs_complex, self.device)

        # Expand keys and values to align with query groups
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Compute attention scores and apply causal mask
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalize attention scores and compute weighted sum of values
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads and apply final projection
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
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
        # Apply attention with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x.to(torch.bfloat16))
        x = x + shortcut
        
        # Apply feedforward network with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))
        x = x + shortcut
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
        # Convert token indices to embeddings
        tok_emb = self.token_embedding(x)
        x = tok_emb
        
        # Pass through transformer layers
        x = self.trf_blocks(tok_emb)
        
        # Normalize and project to logits
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        return logits
