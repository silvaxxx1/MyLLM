import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the Config class from the configuration module
from config import Config

class LayerNorm(nn.Module):
    """Standard LayerNorm implementation with optional bias."""
    
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization, used in LLaMA models."""
    
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Calculate RMS
        norm_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Apply scaling
        return self.weight * x * norm_x


def create_norm_layer(norm_class_name: str, dim: int, eps: float = 1e-5, bias: bool = True):
    """Factory function to create the appropriate normalization layer based on config."""
    if norm_class_name == "LayerNorm":
        return LayerNorm(dim, eps, bias)
    elif norm_class_name == "RMSNorm":
        return RMSNorm(dim, eps, bias)
    else:
        raise ValueError(f"Unsupported normalization layer: {norm_class_name}")


def get_activation_fn(activation: str):
    """Get activation function by name."""
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "silu" or activation == "swish":
        return F.silu
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings used in LLaMA models."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))
        self._update_cos_sin_cache(max_seq_len)
        
    def _update_cos_sin_cache(self, seq_len: int):
        """Update the cached cos and sin values for sequence positions."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Cache cos and sin values
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
    
    def forward(self, x, seq_len: int = None):
        """Apply rotary embeddings to input tensor x."""
        if seq_len is None:
            seq_len = x.size(1)
            
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._update_cos_sin_cache(seq_len)
            
        # Get the appropriate cos and sin values for the current sequence length
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        return cos, sin


def apply_rotary_embeddings(q, k, cos, sin):
    """Apply rotary embeddings to query and key tensors."""
    # Reshape q and k for easier rotation
    q_embed = q.float()
    k_embed = k.float()
    
    # Handle the case where the last dimension is odd
    if q.shape[-1] % 2 != 0:
        q_embed = torch.cat([q_embed, torch.zeros_like(q_embed[..., :1])], dim=-1)
        k_embed = torch.cat([k_embed, torch.zeros_like(k_embed[..., :1])], dim=-1)
    
    # Reshape to split the last dimension into pairs
    q_embed = q_embed.reshape(*q.shape[:-1], -1, 2)
    k_embed = k_embed.reshape(*k.shape[:-1], -1, 2)
    
    # Correct rotation: stack [y, -x] instead of [-y, x]
    q_rot = torch.stack([q_embed[..., 1], -q_embed[..., 0]], dim=-1)
    k_rot = torch.stack([k_embed[..., 1], -k_embed[..., 0]], dim=-1)
    
    # Reshape back
    q_rot = q_rot.reshape(*q.shape[:-1], -1)
    k_rot = k_rot.reshape(*k.shape[:-1], -1)
    
    # Apply rotation using the cos and sin values
    q_embed = q_embed.reshape(*q.shape[:-1], -1)
    k_embed = k_embed.reshape(*k.shape[:-1], -1)
    
    # Truncate if necessary to match the original shape
    q_rot = q_rot[..., :q.shape[-1]]
    k_rot = k_rot[..., :k.shape[-1]]
    q_embed = q_embed[..., :q.shape[-1]]
    k_embed = k_embed[..., :k.shape[-1]]
    
    # Apply the rotation with corrected signs
    q = q_embed * cos - q_rot * sin
    k = k_embed * cos - k_rot * sin
    
    return q.type_as(q), k.type_as(k)


class FlashCausalSelfAttention(nn.Module):
    """
    Optimized causal self-attention using Flash Attention algorithm for better memory efficiency
    and faster computation, especially with longer sequences.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key attention parameters
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        
        # Whether to use rotary embeddings (LLaMA models)
        self.use_rotary = config.rotary_percentage > 0
        self.rotary_dim = int(self.head_dim * config.rotary_percentage)
        
        # Linear projections for Q, K, V, and output
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Initialize rotary embeddings if needed
        if self.use_rotary:
            self.rotary_emb = RotaryEmbedding(dim=self.rotary_dim, max_seq_len=config.block_size)
        
        # Dropout for regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flag to determine if we can use the PyTorch 2.0+ Flash Attention implementation
        self.use_flash_attn = hasattr(F, 'scaled_dot_product_attention')
        
        # If not using Flash Attention, create a causal mask buffer
        if not self.use_flash_attn:
            self.register_buffer(
                "mask", 
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Project input to query, key, and value
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Apply rotary embeddings if used
        if self.use_rotary and self.rotary_dim > 0:
            cos, sin = self.rotary_emb(v, seq_len=seq_len)
            # Apply to the rotary dimensions only
            q_rot, k_rot = q[..., :self.rotary_dim], k[..., :self.rotary_dim]
            q_pass, k_pass = q[..., self.rotary_dim:], k[..., self.rotary_dim:]
            q_rot, k_rot = apply_rotary_embeddings(q_rot, k_rot, cos, sin)
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)
        
        # Use Flash Attention if available
        if self.use_flash_attn:
            # Scale query for numerical stability
            q = q * (1.0 / math.sqrt(self.head_dim))
            # Use PyTorch's optimized attention implementation
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,  # Not needed when is_causal=True
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Fallback implementation for older PyTorch versions
            # Compute attention scores
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # Apply causal mask to prevent attending to future tokens
            mask = self.mask[:, :, :seq_len, :seq_len]
            att = att.masked_fill(mask == 0, float('-inf'))
            
            # Apply softmax and dropout
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            # Weight the values
            y = att @ v
        
        # Combine heads and apply output projection
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class GptMLP(nn.Module):
    """MLP block used in GPT-2 models."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, int(config.mlp_ratio * config.n_embd), bias=config.bias)
        self.c_proj = nn.Linear(int(config.mlp_ratio * config.n_embd), config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.activation)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LLaMAMLP(nn.Module):
    """MLP block used in LLaMA models."""
    
    def __init__(self, config: Config):
        super().__init__()
        hidden_dim = int(config.mlp_ratio * config.n_embd)
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        # LLaMA uses SiLU activation
        self.activation = F.silu

    def forward(self, x):
        # SwiGLU activation
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


def create_mlp_layer(mlp_class_name: str, config: Config):
    """Factory function to create the appropriate MLP layer based on config."""
    if mlp_class_name == "GptMLP":
        return GptMLP(config)
    elif mlp_class_name == "LLaMAMLP":
        return LLaMAMLP(config)
    else:
        raise ValueError(f"Unsupported MLP class: {mlp_class_name}")


class Block(nn.Module):
    """Transformer block with support for different model architectures."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.parallel_residual = config.parallel_residual
        
        # Normalization layers
        self.ln_1 = create_norm_layer(
            config.norm_class_name, 
            config.n_embd, 
            eps=config.norm_eps if config.norm_class_name == "RMSNorm" else config.eps,
            bias=config.bias
        )
        self.ln_2 = create_norm_layer(
            config.norm_class_name, 
            config.n_embd, 
            eps=config.norm_eps if config.norm_class_name == "RMSNorm" else config.eps,
            bias=config.bias
        )
        
        # Attention and MLP
        self.attn = FlashCausalSelfAttention(config)
        self.mlp = create_mlp_layer(config.mlp_class_name, config)

    def forward(self, x):
        if self.parallel_residual:
            # LLaMA-style parallel residual connections
            h = x + self.attn(self.ln_1(x))
            out = h + self.mlp(self.ln_2(h))
        else:
            # GPT-style sequential residual connections
            h = x + self.attn(self.ln_1(x))
            out = h + self.mlp(self.ln_2(h))
        return out


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings used in GPT-2."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, position_ids):
        # Select the appropriate positional embeddings
        pos_emb = self.pos_emb[:, position_ids, :]
        return self.dropout(pos_emb)


# KV Cache implementation for faster generation
class KeyValueCache:
    """Cache for storing key and value tensors during generation."""
    
    def __init__(self, max_batch_size, max_seq_len, n_layers, n_heads, head_dim):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        # Initialize empty cache
        self.k_cache = [None] * n_layers
        self.v_cache = [None] * n_layers
        
        # Cache is not initialized until first use
        self.initialized = False
    
    def initialize(self, batch_size, device, dtype=torch.float16):
        """Initialize the cache with zeros."""
        if self.initialized and batch_size <= self.max_batch_size:
            # Already initialized with sufficient capacity
            return
            
        # Initialize caches for all layers
        for i in range(self.n_layers):
            self.k_cache[i] = torch.zeros(
                (batch_size, self.n_heads, self.max_seq_len, self.head_dim),
                device=device, dtype=dtype
            )
            self.v_cache[i] = torch.zeros(
                (batch_size, self.n_heads, self.max_seq_len, self.head_dim),
                device=device, dtype=dtype
            )
        
        self.initialized = True
    
    def update(self, layer_idx, position_idx, k, v):
        """Update the cache at a specific layer and position."""
        # Update key and value caches
        self.k_cache[layer_idx][:, :, position_idx:position_idx + k.size(2), :] = k
        self.v_cache[layer_idx][:, :, position_idx:position_idx + v.size(2), :] = v
    
    def get(self, layer_idx, position_idx=None):
        """Get cached keys and values up to position_idx."""
        if position_idx is None:
            return self.k_cache[layer_idx], self.v_cache[layer_idx]
        else:
            return (
                self.k_cache[layer_idx][:, :, :position_idx, :],
                self.v_cache[layer_idx][:, :, :position_idx, :]
            )


# Modified attention with KV cache support for efficient generation
class FlashAttentionWithKVCache(FlashCausalSelfAttention):
    """Extended Flash Attention with KV cache support for efficient autoregressive generation."""
    
    def forward(self, x, kv_cache=None, position_ids=None):
        # Handle input dimensions properly
        *dims, _ = x.size()
        batch_size = dims[0] if len(dims) > 0 else 1
        seq_len = dims[1] if len(dims) > 1 else 1

        # Project input to query, key, and value
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if used
        if self.use_rotary and self.rotary_dim > 0:
            # For generation, we need to use absolute positions for RoPE
            if position_ids is not None:
                # Get cos/sin for the positions we need
                cos, sin = self.rotary_emb(None, seq_len=position_ids.max().item() + 1)
                # Select the cos/sin for the positions we're processing
                cos = cos[position_ids].unsqueeze(1)
                sin = sin[position_ids].unsqueeze(1)
            else:
                cos, sin = self.rotary_emb(None, seq_len=seq_len)
                
            # Apply to the rotary dimensions only
            q_rot, k_rot = q[..., :self.rotary_dim], k[..., :self.rotary_dim]
            q_pass, k_pass = q[..., self.rotary_dim:], k[..., self.rotary_dim:]
            q_rot, k_rot = apply_rotary_embeddings(q_rot, k_rot, cos, sin)
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)
        
        # Use KV cache for generation if provided
        if kv_cache is not None:
            layer_idx = kv_cache.get('layer_idx', 0)
            past_length = kv_cache.get('past_length', 0)
            
            # Update KV cache
            kv_cache['k'][layer_idx][:, :, past_length:past_length + seq_len, :] = k
            kv_cache['v'][layer_idx][:, :, past_length:past_length + seq_len, :] = v
            
            # Get full key and value from cache
            k = kv_cache['k'][layer_idx][:, :, :past_length + seq_len, :]
            v = kv_cache['v'][layer_idx][:, :, :past_length + seq_len, :]
            
            # Update past length
            kv_cache['past_length'] = past_length + seq_len
        
        # Use Flash Attention if available
        if self.use_flash_attn:
            # Scale query for numerical stability
            q = q * (1.0 / math.sqrt(self.head_dim))
            
            # For cached implementation, create causal mask
            if kv_cache is not None:
                full_seq_len = k.size(2)
                attention_mask = torch.triu(
                    torch.ones(full_seq_len, full_seq_len, device=q.device, dtype=torch.bool),
                    diagonal=1
                )
                if position_ids is not None:
                    seq_row_offset = position_ids.reshape(-1, seq_len)[:, -1].unsqueeze(-1)
                    seq_col_offset = position_ids.reshape(-1, seq_len).unsqueeze(-2)
                    seq_mask = (seq_col_offset + seq_row_offset.transpose(-2, -1)) >= full_seq_len
                    attention_mask = attention_mask | seq_mask
                
                attention_mask = attention_mask.to(torch.float) * -1e9
                
                y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=attention_mask,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=False
                )
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True
                )
        else:
            # Fallback implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            if kv_cache is None:
                mask = self.mask[:, :, :seq_len, :seq_len]
                att = att.masked_fill(mask == 0, float('-inf'))
            else:
                full_seq_len = k.size(2)
                mask = torch.tril(torch.ones(full_seq_len, full_seq_len, device=att.device))
                att = att.masked_fill(mask == 0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Combine heads and apply output projection
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        y = self.resid_dropout(self.c_proj(y))
        
        return y


# Modified Block with KV cache support
class BlockWithKVCache(Block):
    """Transformer block with KV cache support for efficient generation."""
    
    def __init__(self, config: Config):
        super().__init__(config)  # Fixed super() initialization
        self.parallel_residual = config.parallel_residual
        
        # Re-initialize with correct attention module
        self.attn = FlashAttentionWithKVCache(config)

    def forward(self, x, kv_cache=None, position_ids=None):
        if kv_cache is not None:
            layer_idx = kv_cache.get('layer_idx', 0)
            
            if self.parallel_residual:
                h = x + self.attn(self.ln_1(x), kv_cache=kv_cache, position_ids=position_ids)
                out = h + self.mlp(self.ln_2(h))
            else:
                h = x + self.attn(self.ln_1(x), kv_cache=kv_cache, position_ids=position_ids)
                out = h + self.mlp(self.ln_2(h))
                
            kv_cache['layer_idx'] = layer_idx + 1
            return out
        return super().forward(x)


class Transformer(nn.Module):
    """Unified transformer implementation supporting GPT-2, LLaMA2, and LLaMA3 architectures."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.transformer = nn.ModuleDict()
        self.transformer.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        
        # Positional embeddings
        if config.rotary_percentage == 0:
            self.transformer.wpe = PositionalEmbedding(config)
        else:
            self.transformer.wpe = None
        
        self.emb_scale = math.sqrt(config.n_embd) if config.scale_embeddings else 1.0
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        if hasattr(config, 'use_kv_cache') and config.use_kv_cache:
            self.transformer.h = nn.ModuleList([BlockWithKVCache(config) for _ in range(config.n_layer)])
        else:
            self.transformer.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.transformer.ln_f = create_norm_layer(
            config.norm_class_name, 
            config.n_embd,
            eps=config.norm_eps if config.norm_class_name == "RMSNorm" else config.eps,
            bias=config.bias
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        
        if config.vocab_size == config.padded_vocab_size:
            self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, position_ids=None, kv_cache=None):
        batch_size, seq_len = idx.size()
        token_emb = self.transformer.wte(idx)
        x = token_emb * self.emb_scale
        
        if self.transformer.wpe is not None:
            if position_ids is None:
                if kv_cache is not None:
                    past_length = kv_cache.get('past_length', 0)
                    position_ids = torch.arange(past_length, past_length + seq_len, 
                                               dtype=torch.long, device=idx.device).unsqueeze(0)
                else:
                    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=idx.device).unsqueeze(0).expand(batch_size, -1)
            
            pos_emb = self.transformer.wpe(position_ids)
            x = x + pos_emb
        
        x = self.drop(x)
        
        if kv_cache is not None:
            kv_cache['layer_idx'] = 0
            for block in self.transformer.h:
                x = block(x, kv_cache=kv_cache, position_ids=position_ids)
        else:
            for block in self.transformer.h:
                x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None, use_kv_cache=True):
        kv_cache = None
        if use_kv_cache and hasattr(self.config, 'use_kv_cache') and self.config.use_kv_cache:
            batch_size = idx.size(0)
            device = idx.device
            kv_cache = {
                'k': [torch.zeros(
                    (batch_size, self.config.n_head, self.config.block_size, self.config.n_embd // self.config.n_head),
                    device=device, dtype=next(self.parameters()).dtype
                ) for _ in range(self.config.n_layer)],
                'v': [torch.zeros(
                    (batch_size, self.config.n_head, self.config.block_size, self.config.n_embd // self.config.n_head),
                    device=device, dtype=next(self.parameters()).dtype
                ) for _ in range(self.config.n_layer)],
                'past_length': 0,
                'layer_idx': 0
            }
        
        input_idx = idx
        for _ in range(max_new_tokens):
            if kv_cache is not None and kv_cache['past_length'] > 0:
                idx_cond = idx[:, -1:]
                position_ids = torch.full(
                    (batch_size, 1), 
                    kv_cache['past_length'], 
                    dtype=torch.long, 
                    device=idx.device
                )
            else:
                idx_cond = idx
                position_ids = None
            
            logits, _ = self.forward(idx_cond, position_ids=position_ids, kv_cache=kv_cache)
            logits = logits[:, -1, :]
            
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx

