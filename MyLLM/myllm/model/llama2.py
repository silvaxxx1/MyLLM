import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece  # Sentencepiece for tokenization


# Device setup for training (use GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"


# LLAMA2 Configuration Dictionary (for 7B model, can be modified for other sizes)
LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,
    "context_length": 4096,
    "emb_dim": 4096,
    "num_heads": 32,
    "n_layers": 32,
    "hidden_dim": 11008,
    "dropout": 0.1,
    "dtype": torch.bfloat16
}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm


def pre_compute_freq(config, theta: float = 10000.0):
    head_dim = config["emb_dim"] // config["num_heads"]
    context_length = config["context_length"]

    theta_num = torch.arange(0, head_dim // 2).float()
    theta = 1.0 / (theta ** (2 * theta_num / head_dim))

    m = torch.arange(context_length)
    freq = torch.outer(m, theta)
    return torch.polar(torch.ones_like(freq), freq)


def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:
    _, _, seq_len, _ = x.shape
    freqs_complex = freqs_complex[:seq_len]
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(0)

    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_reshape)
    x_rotate = x_complex * freqs_complex
    x_rotate = torch.view_as_real(x_rotate)
    return x_rotate.reshape(*x.shape)


class RopeFlashAttention(nn.Module):
    def __init__(self, config, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.config = config
        self.num_heads = config["num_heads"]
        self.head_dim = d_out // config["num_heads"]
        self.qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias, dtype=config["dtype"])
        self.proj = nn.Linear(d_out, d_out, dtype=config["dtype"])
        self.register_buffer("freqs_complex", pre_compute_freq(config))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        queries = apply_rope(queries, self.freqs_complex)
        keys = apply_rope(keys, self.freqs_complex)

        out = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, is_causal=True
        )

        context_vec = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.head_dim * self.num_heads)
        return self.proj(context_vec)

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


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config["emb_dim"])
        self.atten = RopeFlashAttention(config, config["emb_dim"], config["emb_dim"])
        self.norm2 = RMSNorm(config["emb_dim"])
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.atten(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class llama2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config["vocab_size"], config["emb_dim"], dtype=config["dtype"])
        self.trs_blk = nn.ModuleList([Transformer(config) for _ in range(config["n_layers"])])
        self.norm = RMSNorm(config["emb_dim"])
        self.proj = nn.Linear(config["emb_dim"], config["vocab_size"], dtype=config["dtype"], bias=False)

    def forward(self, x):
        x = self.token_emb(x)
        for blk in self.trs_blk:
            x = blk(x)
        x = self.norm(x)
        logits = self.proj(x)
        return logits



# Use the provided GPT_CONFIG_124 or create a configuration dictionary
config = LLAMA2_CONFIG_7B

# Initialize the model
model = llama2(config).to(device)

# Example input: let's assume we are working with a vocabulary of 50257 tokens
# and the input is a sequence of token IDs
input_ids = torch.randint(0, config['vocab_size'], (1, config['context_length'])).to(device)  # Random input tokens

# Forward pass
output = model(input_ids)

# Check the output shape
print("Output shape:", output.shape)