# Decoder-Only Transformer Model Implementation (GPT-style)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from config import Config

class GPT(nn.Module):
    """A GPT-like transformer model with support for multiple architectures.
    
    Attributes:
        wte (nn.Embedding): Token embeddings
        wpe (nn.Embedding): Position embeddings (if enabled)
        transformer (nn.ModuleDict): Stack of transformer blocks
        ln_f (nn.Module): Final layer norm
        lm_head (nn.Linear): Output projection
    """
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        
        # Validate crucial parameters
        assert hasattr(config, 'padded_vocab_size'), "Config must specify padded_vocab_size"
        assert hasattr(config, 'n_embd'), "Config must specify n_embd"
        if config.use_learned_pos_emb and config.use_rope:
            raise ValueError("Cannot combine learned positional embeddings and RoPE")

        # Embedding layers
        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        if config.use_learned_pos_emb:
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
            
        # Output projection with optional weight tying
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, 
                               bias=config.lm_head_bias)
        if config.tie_weights:
            assert self.lm_head.weight.shape == self.wte.weight.shape, \
                "Embedding and head dimensions must match for weight tying"
            self.lm_head.weight = self.wte.weight

        # Transformer blocks
        self.transformer = nn.ModuleDict({
            f"block_{i}": Block(config, i) for i in range(config.n_layer)
        })
        self.ln_f = config.norm_class(config.n_embd, eps=config.norm_eps)

        # Initialization
        self.apply(self._init_weights)
        if getattr(config, 'init_method', None) == 'trunc_normal':
            self._apply_trunc_normal_init()

    def _init_weights(self, module):
        """Initialize weights for linear and embedding layers."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _apply_trunc_normal_init(self):
        """Apply truncated normal initialization to all parameters."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.size()
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.config.block_size}")

        # Token embeddings
        tok_emb = self.wte(x)
        x = tok_emb

        # Position embeddings
        if self.config.use_learned_pos_emb:
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
            x = x + self.wpe(pos)

        # Transformer blocks
        for block in self.transformer.values():
            x = block(x)

        # Final output
        return self.lm_head(self.ln_f(x))

class Block(nn.Module):
    """Transformer block implementing both parallel and sequential residuals.
    
    Attributes:
        norm1, norm2 (nn.Module): Pre-attention/MLP normalizations
        attn (CausalSelfAttention): Attention module
        mlp (nn.Module): Feed-forward module
    """
    
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        self.config = config
        
        # Normalization layers
        self.norm1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.norm2 = config.norm_class(config.n_embd, eps=config.norm_eps) 
             
        # Attention and MLP
        self.attn = CausalSelfAttention(config, block_idx)
        self.mlp = config.mlp_class(config)
        
        # Post-normalization layers
        self.post_attn_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) 
            if config.post_attention_norm else nn.Identity()
        )
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps)
            if config.post_mlp_norm else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Attention branch
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        attn_out = self.post_attn_norm(attn_out)

        if self.config.parallel_residual:
            # Parallel MLP branch
            mlp_norm = self.norm2(x) if self.norm2 is not None else x
            mlp_out = self.mlp(mlp_norm)
            
            # Combine with scaling
            scale = 1/math.sqrt(2) if self.config.parallel_residual_scale else 1
            x = residual + (attn_out + mlp_out) * scale
        else:
            # Sequential processing
            x = residual + attn_out
            x_norm = self.norm2(x) if self.norm2 is not None else x
            x = x + self.mlp(x_norm)

        return self.post_mlp_norm(x)

class CausalSelfAttention(nn.Module):
    """Multi-head self attention with support for MHA/MQA/GQA and RoPE."""
    
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        self.config = config
        
        # QKV projections
        self.qkv = nn.Linear(
            config.n_embd, 
            (config.n_head + 2*config.n_query_groups) * config.head_size,
            bias=config.attention_bias
        )
        self.proj = nn.Linear(config.n_head * config.head_size, config.n_embd, 
                            bias=config.bias)
        
        # Query/key normalization
        if config.norm_qk:
            self.norm_q = config.norm_class(config.head_size * config.n_head, 
                                           eps=config.norm_eps)
            self.norm_k = config.norm_class(config.head_size * config.n_query_groups,
                                           eps=config.norm_eps)
        
        # RoPE initialization
        if config.use_rope:
            self.register_buffer(
                "freqs_complex",
                self.pre_compute_freq(config),
                persistent=False
            )

    def pre_compute_freq(self, config: Config) -> torch.Tensor:
        """Precompute RoPE frequencies with scaling validation."""
        head_dim = config.n_embd // config.n_head
        context_length = config.block_size
        
        if getattr(config, 'rope_scaling', None):
            required = ["original_max_seq_len", "factor", 
                       "low_freq_factor", "high_freq_factor"]
            assert all(key in config.rope_scaling for key in required), \
                "Missing RoPE scaling parameters"

        theta = 1.0 / (config.rope_base ** (
            torch.arange(0, head_dim//2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(
            torch.arange(context_length, dtype=torch.float32), 
            theta
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        
        # Split QKV
        q_size = self.config.n_head * self.config.head_size
        k_size = v_size = self.config.n_query_groups * self.config.head_size
        q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
        
        # Normalize queries/keys
        if hasattr(self, 'norm_q'):
            q, k = self.norm_q(q), self.norm_k(k)
            
        # Reshape for attention
        q = q.view(B, T, self.config.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.config.n_query_groups, -1).transpose(1, 2)
        v = v.view(B, T, self.config.n_query_groups, -1).transpose(1, 2)
        
        # Apply RoPE
        if self.config.use_rope:
            q, k = apply_rope(q, self.freqs_complex), apply_rope(k, self.freqs_complex)
            
        # Attention computation
        y = self.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(y)

    def scaled_dot_product_attention(self, q, k, v):
        """Attention computation with softcapping and causal mask."""
        # Validate MQA/GQA setup
        if self.config.n_query_groups != self.config.n_head:
            assert self.config.n_head % self.config.n_query_groups == 0, \
                "n_head must be divisible by n_query_groups"
            k = k.repeat_interleave(self.config.n_head//self.config.n_query_groups, dim=1)
            v = v.repeat_interleave(self.config.n_head//self.config.n_query_groups, dim=1)
            
        scale = 1.0 / math.sqrt(self.config.head_size)
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Softcapping
        if self.config.attention_logit_softcapping:
            attn = torch.tanh(attn / self.config.attention_logit_softcapping) \
                 * self.config.attention_logit_softcapping
            
        # Causal mask
        if self.config.causal_attention:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        return attn @ v

class RMSNorm(nn.Module):
    """RMS Normalization with optional unit offset."""
    
    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = False):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_norm * (1 + self.weight) if self.add_unit_offset 
               else x_norm * self.weight).type_as(x)

class GptMLP(nn.Module):
    """GPT-style MLP with activation flexibility."""
    
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)
        self.proj = nn.Linear(config.mlp_hidden_size, config.n_embd, bias=config.bias)
        self.act = self._get_activation(config.activation)
        
        # GELU approximation handling
        self.gelu_approx = getattr(config, 'gelu_approx', 'none')

    def _get_activation(self, name: str):
        if name == 'gelu':
            return lambda x: F.gelu(x, approximate=self.gelu_approx)
        assert hasattr(F, name), f"Unsupported activation: {name}"
        return getattr(F, name)

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))

class LLaMAMLP(nn.Module):
    """LLaMA-style MLP with SiLU gating."""
    
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)
        self.up = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)
        self.down = nn.Linear(config.mlp_hidden_size, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class KVCache(nn.Module):
    """Efficient KV Cache for autoregressive decoding."""
    
    def __init__(self, batch_size: int, max_seq_len: int, 
                num_kv_heads: int, head_dim: int, dtype: torch.dtype):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros(
            (batch_size, num_kv_heads, max_seq_len, head_dim), dtype=dtype), 
            persistent=False)
        self.register_buffer("v_cache", torch.zeros(
            (batch_size, num_kv_heads, max_seq_len, head_dim), dtype=dtype),
            persistent=False)
        self.register_buffer("cache_pos", torch.tensor(0, dtype=torch.long))

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor):
        seq_len = k_val.size(2)
        remaining = self.k_cache.size(2) - self.cache_pos.item()
        assert seq_len <= remaining, f"Cache overflow: {seq_len} > {remaining}"
        
        start = self.cache_pos.item()
        end = start + seq_len
        
        self.k_cache[:, :, start:end] = k_val
        self.v_cache[:, :, start:end] = v_val
        self.cache_pos += seq_len
        
        return self.k_cache[:, :, :end], self.v_cache[:, :, :end]

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos.zero_()

def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to input tensor."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = x_complex * freqs_complex[:x.size(2)].unsqueeze(0).unsqueeze(0)
    return torch.view_as_real(rotated).flatten(-2).type_as(x)