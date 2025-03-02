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
    
    # Compute the rotated versions
    q_rot = torch.stack([-q_embed[..., 1], q_embed[..., 0]], dim=-1)
    k_rot = torch.stack([-k_embed[..., 1], k_embed[..., 0]], dim=-1)
    
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
    
    # Apply the rotation
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
            # For generation, we need to use absolute positions for RoPE
            if position_ids is not None:
                # Get cos/sin for the positions we need
                cos, sin = self.rotary_emb(None, seq_len=position_ids.max().item() + 1)
                # Select the cos/sin for the positions we're processing
                cos = cos[position_ids].unsqueeze(1)  # Add head dimension
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
            
            # For cached implementation, we can't use is_causal=True because we're 
            # attending to all previous tokens, not just within the current segment
            if kv_cache is not None:
                # Create causal mask for the full sequence length
                full_seq_len = k.size(2)
                attention_mask = torch.triu(
                    torch.ones(full_seq_len, full_seq_len, device=q.device, dtype=torch.bool),
                    diagonal=1
                )
                if position_ids is not None:
                    # If using position IDs, we need to offset the mask
                    seq_row_offset = position_ids.reshape(-1, seq_len)[:, -1].unsqueeze(-1)
                    seq_col_offset = position_ids.reshape(-1, seq_len).unsqueeze(-2)
                    seq_mask = (seq_col_offset + seq_row_offset.transpose(-2, -1)) >= full_seq_len
                    attention_mask = attention_mask | seq_mask
                
                # PyTorch's attention implementation uses additive mask
                attention_mask = attention_mask.to(torch.float) * -1e9
                
                y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=attention_mask,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=False
                )
            else:
                # For regular forward pass, we can use is_causal=True
                y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True
                )
        else:
            # Fallback implementation for older PyTorch versions
            # Compute attention scores
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # Apply causal mask if not using cache
            if kv_cache is None:
                mask = self.mask[:, :, :seq_len, :seq_len]
                att = att.masked_fill(mask == 0, float('-inf'))
            else:
                # For cached implementation, create a custom causal mask
                full_seq_len = k.size(2)
                mask = torch.tril(torch.ones(full_seq_len, full_seq_len, device=att.device))
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


# Modified Block with KV cache support
class BlockWithKVCache(Block):
    """Transformer block with KV cache support for efficient generation."""
    
    def __init__(self, config: Config):
        super(Block, self).__init__()
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
        
        # Attention and MLP with KV cache support
        self.attn = FlashAttentionWithKVCache(config)
        self.mlp = create_mlp_layer(config.mlp_class_name, config)

    def forward(self, x, kv_cache=None, position_ids=None):
        if kv_cache is not None:
            # Update layer index in cache
            layer_idx = kv_cache.get('layer_idx', 0)
            
            if self.parallel_residual:
                # LLaMA-style parallel residual connections
                h = x + self.attn(self.ln_1(x), kv_cache=kv_cache, position_ids=position_ids)
                out = h + self.mlp(self.ln_2(h))
            else:
                # GPT-style sequential residual connections
                h = x + self.attn(self.ln_1(x), kv_cache=kv_cache, position_ids=position_ids)
                out = h + self.mlp(self.ln_2(h))
                
            # Increment layer index for next layer
            kv_cache['layer_idx'] = layer_idx + 1
            return out
        else:
            return super().forward(x)


class Transformer(nn.Module):
    """Unified transformer implementation supporting GPT-2, LLaMA2, and LLaMA3 architectures."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.transformer = nn.ModuleDict()
        self.transformer.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        
        # Use positional embeddings for GPT-2, not for LLaMA (which uses rotary)
        if config.rotary_percentage == 0:
            self.transformer.wpe = PositionalEmbedding(config)
        else:
            self.transformer.wpe = None
        
        # Optional embedding scaling (used in some models)
        self.emb_scale = math.sqrt(config.n_embd) if config.scale_embeddings else 1.0
        
        # Dropout after embeddings
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
        
        # Tie weights between token embedding and LM head if using the same vocab size
        if config.vocab_size == config.padded_vocab_size:
            self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, position_ids=None, kv_cache=None):
        batch_size, seq_len = idx.size()
        
        # Get token embeddings
        token_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = token_emb * self.emb_scale
        
        # Apply positional embeddings if used (GPT-style)
        if self.transformer.wpe is not None:
            # Generate position IDs if not provided
            if position_ids is None:
                if kv_cache is not None:
                    # For generation with KV cache, use absolute positions
                    past_length = kv_cache.get('past_length', 0)
                    position_ids = torch.arange(past_length, past_length + seq_len, 
                                               dtype=torch.long, device=idx.device).unsqueeze(0)
                else:
# For regular forward pass, use sequential positions
                    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=idx.device).unsqueeze(0).expand(batch_size, -1)
            
            # Apply positional embeddings
            pos_emb = self.transformer.wpe(position_ids)
            x = x + pos_emb
        
        # Apply embedding dropout
        x = self.drop(x)
        
        # Apply transformer blocks
        if kv_cache is not None:
            # Reset layer index for each forward pass
            kv_cache['layer_idx'] = 0
            
            # Pass through all blocks with KV cache
            for block in self.transformer.h:
                x = block(x, kv_cache=kv_cache, position_ids=position_ids)
        else:
            # Standard forward pass through all blocks
            for block in self.transformer.h:
                x = block(x)
        
        # Apply final layer norm
        x = self.transformer.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
            
        return logits, loss
    
    def generate(
        self, 
        idx, 
        max_new_tokens=100, 
        temperature=1.0, 
        top_k=None, 
        top_p=None,
        use_kv_cache=True
    ):
        """Generate tokens auto-regressively.
        
        Args:
            idx (torch.Tensor): Context tokens of shape (B, T)
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
            top_k (int): Sample from top-k most likely tokens, or None to disable
            top_p (float): Sample from tokens with cumulative probability < top_p, or None to disable
            use_kv_cache (bool): Whether to use the KV cache for efficient generation
        
        Returns:
            torch.Tensor: Generated tokens of shape (B, T+max_new_tokens)
        """
        # Initialize the KV cache if using it
        kv_cache = None
        if use_kv_cache and hasattr(self.config, 'use_kv_cache') and self.config.use_kv_cache:
            batch_size = idx.size(0)
            device = idx.device
            
            # Create KV cache dictionary
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
        
        # Save original tensor for concatenation later
        input_idx = idx
        
        # Generate new tokens one at a time
        for _ in range(max_new_tokens):
            # If using KV cache, only process the new token(s)
            if kv_cache is not None and kv_cache['past_length'] > 0:
                # During generation, only process the last token
                idx_cond = idx[:, -1:]
                
                # Create position_ids for this step
                position_ids = torch.full(
                    (batch_size, 1), 
                    kv_cache['past_length'], 
                    dtype=torch.long, 
                    device=idx.device
                )
            else:
                # For first step, process all tokens
                idx_cond = idx
                position_ids = None
            
            # Forward pass
            logits, _ = self.forward(idx_cond, position_ids=position_ids, kv_cache=kv_cache)
            
            # Get logits for the last token only
            logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep the first token above the threshold to ensure non-empty output
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted indices back to original shape
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the new token to the context
            idx = torch.cat([idx, next_token], dim=1)
        
        # Return the original context plus the new tokens
        return idx


class ConfigLLaMA(Config):
    """Configuration class for LLaMA models."""
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=11008,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        **kwargs
    ):
        super().__init__(
            n_embd=hidden_size,
            n_head=num_attention_heads,
            n_layer=num_hidden_layers,
            block_size=max_position_embeddings,
            bias=False,
            vocab_size=vocab_size,
            padded_vocab_size=vocab_size,  # No padding for LLaMA vocab
            dropout=0.0,  # LLaMA doesn't use dropout
            activation="silu",
            norm_eps=rms_norm_eps,
            **kwargs
        )
        
        # LLaMA-specific settings
        self.norm_class_name = "RMSNorm"
        self.norm_eps = rms_norm_eps
        self.mlp_class_name = "LLaMAMLP"
        self.mlp_ratio = 8/3  # Equivalent to intermediate_size/hidden_size
        self.rotary_percentage = 1.0  # LLaMA uses full rotary embeddings
        self.parallel_residual = True  # LLaMA uses parallel residual connections
        self.scale_embeddings = False  # LLaMA doesn't scale embeddings
        
        # Token IDs
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # For efficient generation
        self.use_kv_cache = use_cache


class ConfigGPT2(Config):
    """Configuration class for GPT-2 models."""
    
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        mlp_ratio=4,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        **kwargs
    ):
        super().__init__(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=n_positions,
            bias=True,
            vocab_size=vocab_size,
            padded_vocab_size=vocab_size,  # GPT-2 doesn't use padding for vocab
            dropout=resid_pdrop,
            activation=activation_function,
            norm_eps=layer_norm_epsilon,
            **kwargs
        )
        
        # GPT-2-specific settings
        self.norm_class_name = "LayerNorm"
        self.mlp_class_name = "GptMLP"
        self.mlp_ratio = mlp_ratio
        self.rotary_percentage = 0.0  # GPT-2 uses traditional positional embeddings
        self.parallel_residual = False  # GPT-2 uses sequential residual connections
        self.scale_embeddings = True  # GPT-2 scales embeddings
        
        # Additional dropout rates
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        
        # For efficient generation
        self.use_kv_cache = True


def main():
    """Example usage of the model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or sample from transformer model')
    parser.add_argument('--model', type=str, default='gpt2', help='Model architecture (gpt2 or llama)')
    parser.add_argument('--train', action='store_true', help='Train the model instead of sampling')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create configuration
    if args.model.lower() == 'gpt2':
        config = ConfigGPT2()
    elif args.model.lower() == 'llama':
        config = ConfigLLaMA()
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Create model
    model = Transformer(config).to(args.device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    if args.train:
        # Training logic would go here
        print("Training not implemented in this example")
    else:
        # Generate text
        # Create a sample input (batch_size, 1) with BOS token
        bos_token_id = getattr(config, 'bos_token_id', 0)
        input_ids = torch.full((args.batch_size, 1), bos_token_id, dtype=torch.long, device=args.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
        
        # In a real application, you would decode the generated IDs to text
        print(f"Generated sequence shape: {generated_ids.shape}")


if __name__ == "__main__":
    main() 