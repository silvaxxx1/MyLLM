import torch 

# Llama 3.2 1B

LLAMA32_CONFIG = {
    "vocab_size": 128_256,      # Vocabulary size
    "context_length": 131_072,  # Context length
    "emb_dim": 2048,            # Embedding dimension
    "n_heads": 32,              # Number of attention heads
    "n_layers": 16,             # Number of layers
    "hidden_dim": 8192,         # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,     # The base in RoPE's "theta"
    "dtype": torch.bfloat16,    # Lower-precision dtype to reduce memory usage
    "rope_freq": {              # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

# Llama 3.2 3B

# LLAMA32_CONFIG = {
#     "vocab_size": 128_256,      # Vocabulary size
#     "context_length": 131_072,  # Context length
#     "emb_dim": 3072,            # Embedding dimension
#     "n_heads": 24,              # Number of attention heads
#     "n_layers": 28,             # Number of layers
#     "hidden_dim": 8192,         # Size of the intermediate dimension in FeedForward
#     "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
#     "rope_base": 500_000.0,     # The base in RoPE's "theta"
#     "dtype": torch.bfloat16,    # Lower-precision dtype to reduce memory usage
#     "rope_freq": {              # RoPE frequency scaling
#         "factor": 32.0,
#         "low_freq_factor": 1.0,
#         "high_freq_factor": 4.0,
#         "original_context_length": 8192,
#     }
# }

LLAMA_SIZE_STR = "1B" if LLAMA32_CONFIG["emb_dim"] == 2048 else "3B"