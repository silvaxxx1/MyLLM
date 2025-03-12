import torch 
import torch.nn as nn 
import torch.nn.functional as F  


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

class Llama3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryFlashAttention(
            config=cfg,  # Pass the config dictionary
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            qkv_bias=False
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x.to(torch.bfloat16))   # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))
        x = x + shortcut  # Add the original input back

        return x

def pre_compute_freq(config, theta: float = 10000.0):
    if config.get("rope_freq") is not None:
        # Advanced RoPE (Llama3) computation using wavelength adjustments.
        head_dim = config["emb_dim"] // config["n_heads"]
        context_length = config["context_length"]

        # Base inverse frequency.
        theta_idx = torch.arange(0, head_dim // 2, dtype=torch.float32)
        inv_freq = 1.0 / (config["rope_base"] ** (2 * theta_idx / head_dim))
        
        # Wavelength thresholds.
        low_wavelength = config["rope_freq"]["original_context_length"] / config["rope_freq"]["low_freq_factor"]
        high_wavelength = config["rope_freq"]["original_context_length"] / config["rope_freq"]["high_freq_factor"]
        
        # Compute wavelengths for each frequency.
        wavelen = 2 * torch.pi / inv_freq
        
        # Adjust frequencies: if wavelength exceeds the low threshold, scale it.
        inv_freq_adj = torch.where(wavelen > low_wavelength,
                                   inv_freq / config["rope_freq"]["factor"],
                                   inv_freq)
        
        # Compute a smooth factor for medium frequencies.
        smooth_factor = ((config["rope_freq"]["original_context_length"] / wavelen) -
                         config["rope_freq"]["low_freq_factor"]) / (
                            config["rope_freq"]["high_freq_factor"] -
                            config["rope_freq"]["low_freq_factor"])
        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / config["rope_freq"]["factor"]) + smooth_factor * inv_freq
        
        # Use smoothed frequencies for medium-range wavelengths.
        is_medium = (wavelen <= low_wavelength) & (wavelen >= high_wavelength)
        inv_freq = torch.where(is_medium, smoothed_inv_freq, inv_freq_adj)
        
        # Compute final frequency matrix.
        positions = torch.arange(context_length, dtype=torch.float32)
        freq = torch.outer(positions, inv_freq)
        return torch.polar(torch.ones_like(freq), freq)
    
    else:
        # Fallback: original RoPE computation.
        num_heads = config.get("num_heads", config.get("n_heads"))
        head_dim = config["emb_dim"] // num_heads
        context_length = config["context_length"]
        theta_idx = torch.arange(0, head_dim // 2, dtype=torch.float32)
        theta_val = 1.0 / (theta ** (2 * theta_idx / head_dim))
        positions = torch.arange(context_length, dtype=torch.float32)
        freq = torch.outer(positions, theta_val)
        return torch.polar(torch.ones_like(freq), freq)
    
    
def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:
    _, _, seq_len, _ = x.shape
    freqs_complex = freqs_complex[:seq_len]
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(0)

    # Cast input to float32 for view_as_complex
    x_reshape = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_reshape)
    x_rotate = x_complex * freqs_complex
    x_rotate = torch.view_as_real(x_rotate)

    # Cast back to the original dtype (bfloat16)
    return x_rotate.reshape(*x.shape).to(x.dtype)

class GroupedQueryFlashAttention(nn.Module):
    def __init__(self, config, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.config = config
        self.n_heads = config["n_heads"]
        self.n_kv_groups = config["n_kv_groups"]
        self.head_dim = d_out // self.n_heads
        
        # Correct QKV projection dimensions
        self.q_proj = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=config["dtype"])
        self.k_proj = nn.Linear(d_in, self.n_kv_groups * self.head_dim, bias=qkv_bias, dtype=config["dtype"])
        self.v_proj = nn.Linear(d_in, self.n_kv_groups * self.head_dim, bias=qkv_bias, dtype=config["dtype"])
        self.proj = nn.Linear(d_out, d_out, dtype=config["dtype"])
        
        # Precompute frequencies
        self.register_buffer("freqs_complex", pre_compute_freq(config))
    
    def forward(self, x):
        B, T, _ = x.shape
        # Project Q, K, V separately
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k = self.k_proj(x).reshape(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)  # [B, g, T, hd]
        v = self.v_proj(x).reshape(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)  # [B, g, T, hd]
        
        # Apply RoPE to Q and K
        q = apply_rope(q, self.freqs_complex)
        k = apply_rope(k, self.freqs_complex)
        
        # Expand K and V to match the number of query heads per group
        k = k[:, :, None, :, :].expand(-1, -1, self.n_heads // self.n_kv_groups, -1, -1)
        v = v[:, :, None, :, :].expand(-1, -1, self.n_heads // self.n_kv_groups, -1, -1)
        k = k.reshape(B, self.n_heads, T, self.head_dim)
        v = v.reshape(B, self.n_heads, T, self.head_dim)
        
        # Use Flash Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        return self.proj(attn_out)
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)



# Initialize the model with the provided configuration
model = Llama3(LLAMA32_CONFIG)

# Generate a dummy input tensor
# Assuming batch size of 2 and sequence length of 10
batch_size = 2
seq_length = 10
dummy_input = torch.randint(0, LLAMA32_CONFIG["vocab_size"], (batch_size, seq_length))

# Forward pass through the model
logits = model(dummy_input)

# Check the output shape
print(logits.shape)  # Expected shape: [batch_size, seq_length, vocab_size]


    




