import torch
import torch.nn as nn
import torch.nn.functional as F 

from dataclasses import dataclass

@dataclass
class AttentionConfig:
    attention_type: str          # "mha", "mqa", or "gqa"
    embed_dim: int               # Total embedding dimension (e.g., 256)
    query_heads: int = 8         # Number of query heads
    kv_heads: int = 8            # Number of key/value heads
    head_dim: int = 64           # Dimension per head

    def __post_init__(self):
        assert self.attention_type in {"mha", "mqa", "gqa"}, \
            f"Invalid attention_type: {self.attention_type}"

        # Check embed_dim compatibility
        assert self.query_heads * self.head_dim == self.embed_dim, \
            f"embed_dim must equal query_heads * head_dim, got {self.embed_dim} ≠ {self.query_heads}×{self.head_dim}"

        # Logic per attention type
        if self.attention_type == "mqa":
            assert self.kv_heads == 1, "MQA requires kv_heads=1"
        elif self.attention_type == "gqa":
            assert self.kv_heads < self.query_heads, \
                f"GQA requires kv_heads < query_heads (got {self.kv_heads} ≥ {self.query_heads})"
            assert self.query_heads % self.kv_heads == 0, \
                "query_heads must be divisible by kv_heads in GQA"
        elif self.attention_type == "mha":
            assert self.kv_heads == self.query_heads, \
                "MHA requires kv_heads == query_heads"

class CausalSelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config


