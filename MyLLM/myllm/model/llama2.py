import torch
import torch.nn as nn
import torch.nn.functional as F

import sentencepiece  # Sentencepiece for tokenization

# Device setup for training (use GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# LLAMA2 Configuration Dictionary (for 7B model)
LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length (number of tokens the model can process at once)
    "emb_dim": 4096,         # Embedding dimension (size of token embeddings)
    "num_heads": 32,         # Number of attention heads in each layer
    "n_layers": 32,          # Number of layers in the Transformer
    "hidden_dim": 11008,     # Size of the intermediate dimension in the FeedForward layers
    "dtype": torch.bfloat16  # Lower-precision dtype to reduce memory usage (bfloat16 for faster training)
}

# Main Model Class: llama2
class llama2(nn.Module):
    def __init__(self, config):
        """
        Initialize the llama2 model with the given configuration.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super().__init__()
        # Embedding layer to map tokens to vectors
        self.token_emb = nn.Embedding(config["vocab_size"], config["emb_dim"], dtype=config["dtype"]) 
        
        # List of Transformer blocks (layers)
        self.trs_blk = nn.ModuleList([Transformer(config) for _ in range(config["n_layers"])])
        
        # Layer normalization after all transformer layers
        self.norm = nn.RMSNorm(config["emb_dim"]) 
        
        # Final linear projection to output logits over the vocabulary
        self.proj = nn.Linear(config["emb_dim"], config["vocab_size"], dtype=config["dtype"], bias=False)

    def forward(self, x):
        """
        Forward pass through the llama2 model.

        Args:
            x (torch.Tensor): Input token indices of shape (batch_size, seq_len).
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size).
        """
        x = self.token_emb(x)  # Token embeddings
        for blk in self.trs_blk:
            x = blk(x)  # Pass through each Transformer block
        
        x = self.norm(x)  # Apply layer normalization
        logits = self.proj(x)  # Project to vocab size

        return logits


def pre_compute_freq(config, theta: float = 10000.0):
    """
    Precompute rotary positional encoding frequencies.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        theta (float): Scaling factor for positional frequencies.
    
    Returns:
        torch.Tensor: Tensor of complex numbers for positional encodings.
    """
    head_dim = config["emb_dim"] // config["num_heads"]  # Use head dimension
    context_length = config["context_length"]
    
    # Create a tensor for theta values based on head dimension
    theta_num = torch.arange(0, head_dim // 2).float()  # Based on head_dim, not num_heads
    theta = 1.0 / (theta ** (2 * theta_num / head_dim))  # Key fix for positional frequencies
    
    m = torch.arange(context_length)  # Context length (sequence length)
    freq = torch.outer(m, theta)  # Create a tensor for frequency values

    # Return a complex tensor with polar representation (cosine, sine)
    return torch.polar(torch.ones_like(freq), freq)


def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional encoding (RoPE) to input tensor.

    Args:
        x (torch.Tensor): Input tensor (batch_size, seq_len, num_heads, head_dim).
        freqs_complex (torch.Tensor): Precomputed complex frequencies.

    Returns:
        torch.Tensor: Tensor with RoPE applied.
    """
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)  # Reshape for complex view
    x_complex = torch.view_as_complex(x_reshape)
    x_rotate = x_complex * freqs_complex  # Apply rotation
    x_rotate = torch.view_as_real(x_rotate)  # Convert back to real values
    x_out = x_rotate.reshape(*x.shape)  # Reshape back to original tensor shape
    return x_out


class RopeFlashAttention(nn.Module):
    def __init__(self, config, d_in, d_out, qkv_bias=False):
        """
        Initialize the RopeFlashAttention layer.

        Args:
            config (dict): Configuration dictionary containing model parameters.
            d_in (int): Input dimension.
            d_out (int): Output dimension.
            qkv_bias (bool): Whether to use bias in the QKV projection layer.
        """
        super().__init__()
        self.config = config
        assert d_out % config["num_heads"] == 0, "d_out must be divisible by num_heads"
        self.head_dim = d_out // config["num_heads"]
        self.d_out = d_out
        self.qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias, dtype=config["dtype"])
        self.proj = nn.Linear(d_out, d_out, dtype=config["dtype"])

        # Precompute and register the freqs_complex buffer (rotary frequencies)
        self.register_buffer("freqs_complex", pre_compute_freq(config))
        self.apply_rope = apply_rope  # RoPE implementation

    def forward(self, x):
        """
        Forward pass through the RopeFlashAttention layer.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, emb_dim).
        
        Returns:
            torch.Tensor: Output tensor after attention (batch_size, seq_len, d_out).
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V using the qkv projection
        queries, keys, values = self.qkv(x).view(batch_size, seq_len, 3, self.config["num_heads"], self.head_dim).permute(2, 0, 3, 1, 4)

        # Apply RoPE (no need to pass device as freqs_complex is already a buffer)
        queries = self.apply_rope(queries, self.freqs_complex)
        keys = self.apply_rope(keys, self.freqs_complex)

        # Perform scaled dot-product attention
        out = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            is_causal=True
        )

        # Project the output from attention
        context_vec = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)
        return self.proj(context_vec)


class Transformer(nn.Module):
    def __init__(self, config):
        """
        Initialize the Transformer block with RoPE and FeedForward layer.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super().__init__()
        self.norm1 = nn.RMSNorm(config["emb_dim"])  # LayerNorm before attention
        self.atten = RopeFlashAttention(config, config["emb_dim"], config["emb_dim"])  # Attention layer with RoPE
        self.norm2 = nn.RMSNorm(config["emb_dim"])  # LayerNorm before FeedForward
        self.ffn = FeedForward(config)  # FeedForward Network

    def forward(self, x):
        """
        Forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, emb_dim).
        
        Returns:
            torch.Tensor: Output tensor after attention and FeedForward pass.
        """
        # Pre-LN residual structure
        x = x + self.atten(self.norm1(x))  # Residual after attention
        x = x + self.ffn(self.norm2(x))    # Residual after FeedForward
        return x


class FeedForward(nn.Module):
    def __init__(self, cfg):
        """
        Initialize the FeedForward layer for each Transformer block.

        Args:
            cfg (dict): Configuration dictionary containing model parameters.
        """
        super().__init__()
        # FeedForward consists of three linear layers
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        """
        Forward pass through the FeedForward layer.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, emb_dim).
        
        Returns:
            torch.Tensor: Output tensor after FeedForward operations.
        """
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2  # Apply SwiGLU activation
        return self.fc3(x)

