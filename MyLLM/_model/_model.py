# Decoder-Only Transformer Model Implementation (GPT-style)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from config import Config

class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        # Validate incompatible configurations
        if config.use_learned_pos_emb and config.use_rope:
            raise ValueError("Cannot use both learned positional embeddings and RoPE.")
            
        # Embedding layers
        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        if config.use_learned_pos_emb:
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
            
        # Weight tying
        if config.tie_weights:
            self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
            self.lm_head.weight = self.wte.weight
        else:
            self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)

        # Transformer blocks
        self.transformer = nn.ModuleDict(
            {f"block_{i}": Block(config, i) for i in range(config.n_layer)}
        )
        self.ln_f = config.norm_class(config.n_embd, eps=config.norm_eps)

        # Initialize weights
        self.apply(self._init_weights)
        if hasattr(config, 'init_method'):
            self._apply_custom_init(config.init_method)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_custom_init(self, init_method):
        if init_method == 'trunc_normal':
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
            pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.wpe(pos)
            x = x + pos_emb

        # Transformer blocks
        for block in self.transformer.values():
            x = block(x)

        # Final output
        x = self.ln_f(x)
        return self.lm_head(x)

class Block(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        self.config = config
        
        # Normalization layers
        self.norm1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.norm2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        
        # Attention and MLP
        self.attn = CausalSelfAttention(config, block_idx)
        self.mlp = config.mlp_class(config)
        
        # Normalization configurations
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
        x_norm = self.norm1(x)
        
        attn_out = self.attn(x_norm)
        attn_out = self.post_attn_norm(attn_out)
        
        if self.config.parallel_residual:
            # Parallel branch
            mlp_norm = self.norm2(x) if self.norm2 else x
            mlp_out = self.mlp(mlp_norm)
            
            # Scaled residual connection
            scale = 1.0/math.sqrt(2) if self.config.parallel_residual_scale else 1.0
            x = residual + (attn_out + mlp_out) * scale
        else:
            # Sequential processing
            x = residual + attn_out
            x = x + self.mlp(self.norm2(x) if self.norm2 else self.mlp(x)
            
        return self.post_mlp_norm(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        
        # QKV projections
        self.qkv = nn.Linear(
            config.n_embd, 
            (config.n_head + 2*config.n_query_groups) * config.head_size,
            bias=config.attention_bias
        )
        self.proj = nn.Linear(config.n_head * config.head_size, config.n_embd, bias=config.bias)
        
        # Query/key normalization
        if config.norm_qk:
            self.norm_q = config.norm_class(config.head_size * config.n_head, eps=config.norm_eps)
            self.norm_k = config.norm_class(config.head_size * config.n_query_groups, eps=config.norm_eps)
        
        # RoPE initialization
        if config.use_rope:
            self.register_buffer(
                "freqs_complex",
                pre_compute_freq(
                    config=config,
                    context_length=config.block_size,
                    base=config.rope_base
                ),
                persistent=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        
        # Split into query/key/value
        q_size = self.config.n_head * self.config.head_size
        k_size = v_size = self.config.n_query_groups * self.config.head_size
        q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
        
        # Normalize queries/keys if configured
        if hasattr(self, 'norm_q'):
            q = self.norm_q(q)
            k = self.norm_k(k)
            
        # Reshape for attention
        q = q.view(B, T, self.config.n_head, self.config.head_size).transpose(1, 2)
        k = k.view(B, T, self.config.n_query_groups, self.config.head_size).transpose(1, 2)
        v = v.view(B, T, self.config.n_query_groups, self.config.head_size).transpose(1, 2)
        
        # Apply RoPE
        if self.config.use_rope:
            q = apply_rope(q, self.freqs_complex)
            k = apply_rope(k, self.freqs_complex)
            
        # Attention computation
        y = self.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(y)

    def scaled_dot_product_attention(self, q, k, v):
        scale = 1.0 / math.sqrt(self.config.head_size)
        
        # Handle MQA/GQA
        if self.config.n_query_groups != self.config.n_head:
            repeat_factor = self.config.n_head // self.config.n_query_groups
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
            
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Softcapping if configured
        if self.config.attention_logit_softcapping:
            attn = softcapping(attn, self.config.attention_logit_softcapping)
            
        # Causal mask
        if self.config.causal_attention:
            mask = torch.triu(torch.ones(attn.size(-2), attn.size(-1), 
                                       dtype=torch.bool, device=attn.device), diagonal=1)
            attn = attn.masked_fill(mask, float('-inf'))
            
        # Softmax and output
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        return attn @ v

def softcapping(x: torch.Tensor, thresh: float) -> torch.Tensor:
    return torch.tanh(x / thresh) * thresh

def pre_compute_freq(config, context_length=None, base=10000.0, device=None, extra_config=None):
    head_dim = config.n_embd // config.n_head
    context_length = context_length or config.block_size
    
    # Validate RoPE scaling parameters
    if extra_config:
        required = ["original_max_seq_len", "factor", "low_freq_factor", "high_freq_factor"]
        for key in required:
            if key not in extra_config:
                raise ValueError(f"Missing RoPE scaling parameter: {key}")
    
    # Frequency calculation
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    pos = torch.arange(context_length, device=device)
    freqs = torch.outer(pos, theta)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(x, freqs_complex):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = x_complex * freqs_complex
    return torch.view_as_real(rotated).flatten(-2).type_as(x)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps
        if self.add_unit_offset:
            return (x_norm * (1 + self.weight)).type_as(x)
        return (x_norm * self.weight).type_as(x)

class GptMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)
        self.proj = nn.Linear(config.mlp_hidden_size, config.n_embd, bias=config.bias)
        self.act = F.gelu if config.activation == 'gelu' else getattr(F, config.activation)

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))

class LLaMAMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)
        self.up = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)
        self.down = nn.Linear(config.mlp_hidden_size, config.n_embd, bias=config.bias)
        self.act = F.silu

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))

class KVCache(nn.Module):
    def __init__(self, batch_size: int, max_seq_len: int, 
                num_kv_heads: int, head_dim: int, dtype: torch.dtype):
        super().__init__()
        self.register_buffer(
            "k_cache",
            torch.zeros((batch_size, num_kv_heads, max_seq_len, head_dim), dtype=dtype),
            persistent=False
        )
        self.register_buffer(
            "v_cache",
            torch.zeros((batch_size, num_kv_heads, max_seq_len, head_dim), dtype=dtype),
            persistent=False
        )
        self.register_buffer("cache_pos", torch.tensor(0, dtype=torch.long), persistent=False)

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor):
        if k_val.size(0) > self.k_cache.size(0):
            raise ValueError(f"Batch size {k_val.size(0)} exceeds cache size {self.k_cache.size(0)}")
            
        start_pos = self.cache_pos.item()
        end_pos = start_pos + k_val.size(2)
        
        self.k_cache[:, :, start_pos:end_pos] = k_val
        self.v_cache[:, :, start_pos:end_pos] = v_val
        self.cache_pos += k_val.size(2)
        
        return self.k_cache[:, :, :end_pos], self.v_cache[:, :, :end_pos]

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos.zero_() 