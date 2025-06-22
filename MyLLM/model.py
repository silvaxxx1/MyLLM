"""
Decoder-Only Transformer Model Implementation (GPT-style)

This module provides a flexible and scalable implementation of a decoder-only
transformer model, similar to the GPT architecture. The implementation is designed
to support various configurations and architectural variations.

Key Features:
- Multi-head attention with support for:
  * Multi-Head Attention (MHA)
  * Multi-Query Attention (MQA) 
  * Grouped-Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- Key-Value caching for efficient autoregressive inference
- Multiple MLP variants:
  * Standard GPT-style (GELU activation)
  * LLaMA-style (SwiGLU activation)
- Configurable normalization layers:
  * LayerNorm
  * RMSNorm
- Flexible residual connection patterns:
  * Parallel residuals (like LLaMA)
  * Sequential residuals (like GPT)
- Comprehensive attention masking support
- Extensive configuration options

Supported Architectures:
- GPT-2 style models
- GPT-Neo/J style models
- LLaMA 1/2 style models
- Other decoder-only transformer variants

Implementation Details:
- Highly modular design with clear separation of components
- Full type hints for better IDE support and maintainability
- Comprehensive documentation for all classes and methods
- Optimized for both training and inference
- GPU acceleration support through PyTorch

Example Usage:
    >>> from config import Config
    >>> config = Config.from_name("llama2-7b")
    >>> model = GPT(config)
    >>> inputs = torch.randint(0, config.vocab_size, (1, 10))
    >>> outputs = model(inputs)
    >>> print(outputs.shape)  # (1, 10, vocab_size)

Performance Considerations:
- KV caching significantly speeds up autoregressive generation
- RoPE provides better positional information than learned embeddings  
- Parallel residuals can improve training efficiency
- GQA/MQA reduce memory usage during inference

References:
1. "Attention Is All You Need" - Vaswani et al. (2017)
2. "Language Models are Few-Shot Learners" - Brown et al. (2020)
3. "LLaMA: Open and Efficient Foundation Language Models" - Touvron et al. (2023)
"""

import torch  
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Tuple

# Import the configuration class from the config module
from config import Config 

class GPT(nn.Module):
    """
A GPT-like transformer model implementing a decoder-only architecture.

This class implements a flexible transformer model that can be configured
for various architectures (GPT-2, GPT-Neo, LLaMA, etc.). It supports
different attention mechanisms, position embeddings, and model configurations.

Architecture Overview:
┌──────────────────┐
│    Embeddings    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Transformer Block│ × n_layer
│  ┌────────────┐ │
│  │ Attention  │ │
│  └────────────┘ │
│  ┌────────────┐ │
│  │    MLP     │ │
│  └────────────┘ │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Layer Norm (f)  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│    LM Head       │
└──────────────────┘

Key Components:
1. Token and position embeddings
2. Stack of transformer blocks
3. Final layer normalization
4. Language modeling head

Configuration Options:
- Number of layers (n_layer)
- Number of attention heads (n_head)
- Embedding dimension (n_embd)
- Vocabulary size (vocab_size)
- Context length (block_size)
- Attention type (MHA/MQA/GQA)
- Position embedding type (learned/RoPE)
- Normalization type (LayerNorm/RMSNorm)
- Residual connection type (parallel/sequential)

Methods:
- forward: Compute model outputs
- initialize_kv_cache: Prepare for autoregressive generation  
- reset_cache: Clear the KV cache
"""
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        # Validate configuration
        if not hasattr(config, 'padded_vocab_size'):
            raise ValueError("Config must specify 'padded_vocab_size'.")
        if not hasattr(config, 'n_embd'):
            raise ValueError("Config must specify 'n_embd'.")

        # Linear layer to map from embedding size to vocabulary size for output logits
        self.lm_head = nn.Linear(
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias
        )

        # Embedding layer for token IDs
        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        
        # Position embeddings if configured
        if getattr(config, 'position_embedding', 'learned') == "learned":
            self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # Transformer blocks (decoder layers)
        self.transformer = nn.ModuleDict(
            {f"block_{block_idx}": Block(config, block_idx) for block_idx in range(config.n_layer)}
        )

        # Final layer normalization before output
        self.ln_f = config.norm_class(config.n_embd, eps=config.norm_eps)
        
        # Track if KV cache is initialized
        self.kv_cache_initialized = False

    def initialize_kv_cache(self, batch_size: int, max_seq_len: int, dtype=torch.float32) -> None:
        """Initialize the key-value cache for autoregressive generation."""
        head_dim = self.config.n_embd // self.config.n_head
        num_kv_heads = self.config.n_query_groups
        
        # Create a KV cache for each transformer block
        for block in self.transformer.values():
            block.attn.initialize_kv_cache(batch_size, max_seq_len, num_kv_heads, head_dim, dtype)
        
        self.kv_cache_initialized = True

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """
Forward pass through the GPT model.

Parameters:
    x (torch.Tensor): 
        Input tensor of shape (batch_size, seq_len) containing token indices
    use_cache (bool):
        Whether to use KV caching for autoregressive generation

Returns:
    torch.Tensor: 
        Logits of shape (batch_size, seq_len, vocab_size) representing
        unnormalized probabilities for each token in the vocabulary

Processing Steps:
1. Input validation (sequence length check)
2. Token embedding lookup
3. Position embedding addition (if configured)
4. Sequential processing through transformer blocks
5. Final layer normalization
6. Projection to vocabulary space

Notes:
- When use_cache=True, expects to be called sequentially with increasing positions
- Input sequences longer than config.block_size will raise an error
"""
        B, T = x.size()  # B = batch size, T = sequence length

        # Check if input sequence length is within the allowed block size
        if T > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {T} tokens, block size is only {self.config.block_size}."
            )

        # Token embeddings
        token_embeddings = self.wte(x)

        # Add position embeddings if they exist
        if hasattr(self, 'wpe'):
            pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
            position_embeddings = self.wpe(pos)
            x = token_embeddings + position_embeddings
        else:
            x = token_embeddings

        # Pass through each transformer block
        for block in self.transformer.values():
            x = block(x, use_cache=use_cache)

        # Final layer normalization
        x = self.ln_f(x)

        # Output logits
        return self.lm_head(x)
        
    def reset_cache(self) -> None:
        """Reset the KV cache for all transformer blocks."""
        for block in self.transformer.values():
            if hasattr(block.attn, 'kv_cache') and block.attn.kv_cache is not None:
                block.attn.kv_cache.reset()
        self.kv_cache_initialized = False


class Block(nn.Module):
    """
A single transformer block (decoder layer).

Implements the core transformer operations:
1. Self-attention mechanism
2. Position-wise feed-forward network
3. Residual connections
4. Normalization layers

Architecture Variations:

Non-parallel residual:          Parallel residual:
┌─── x                         ┌─── x ─────────────────┐
│    ↓                        │    ↓                   ↓
│  norm_1                     │  norm_1                norm_2
│    ↓                        │    ↓                     ↓
│  attn                       │  attn                   mlp
│    ↓                        │    ↓                     ↓
│    + ←── x                  └──→ + ←───────────────── +
│    ↓                             ↓
│  norm_2                          out
│    ↓
│   mlp
│    ↓
│    + ←── x
│    ↓
└──→ out

Components:
- Attention mechanism (CausalSelfAttention)
- MLP (GptMLP or LLaMAMLP)
- Normalization layers
- Residual connections

Configuration Options:
- Parallel vs sequential residuals
- Shared attention norm
- Post-attention norm
- Post-MLP norm
"""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()

        # Check for unsupported configurations
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )
        
        # Layer Normalization before Attention
        self.norm1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        
        # Optional second layer normalization before MLP
        self.norm2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        
        # Causal self-attention mechanism
        self.attn = CausalSelfAttention(config, block_idx)
        
        # Post-attention normalization
        self.post_attention_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_attention_norm else nn.Identity()
        )
        
        # Post-MLP normalization
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_mlp_norm else nn.Identity()
        )
        
        # Multi-layer Perceptron (MLP) component
        self.mlp = config.mlp_class(config)
        self.config = config

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # Apply the first normalization
        x_normed = self.norm1(x)

        # Apply self-attention with optional KV cache
        attn_out = self.attn(x_normed, use_cache=use_cache)

        # Apply post-attention normalization
        attn_out = self.post_attention_norm(attn_out)

        if self.config.parallel_residual:
            # Shared norm case: use x_normed, otherwise apply norm2
            mlp_norm_input = x_normed if self.norm2 is None else self.norm2(x)
            
            # Apply MLP
            mlp_out = self.mlp(mlp_norm_input)
            mlp_out = self.post_mlp_norm(mlp_out)

            # Sum attention and MLP outputs in parallel
            x = attn_out + mlp_out + x  
        else:
            # Standard residual: add attention output first
            x = x + attn_out

            # Apply second norm if necessary
            x_normed = self.norm2(x) if self.norm2 is not None else x

            # Apply MLP
            mlp_out = self.mlp(x_normed)
            mlp_out = self.post_mlp_norm(mlp_out)

            # Add MLP output to the running sum
            x = x + mlp_out

        return x


class CausalSelfAttention(nn.Module):
    """
Causal self-attention mechanism supporting various attention patterns.

Implements different attention mechanisms:
- Multi-Head Attention (MHA)
- Multi-Query Attention (MQA) 
- Grouped-Query Attention (GQA)

Attention Patterns:

Standard MHA:                    GQA (n_query_groups=2):
q k v   q k v   q k v          q q k v   q q k v
│ │ │   │ │ │   │ │ │          │ │ │ │   │ │ │ │
└─┼─┼───┼─┼─┼───┼─┼─┘          └─┼─┼─┼───┼─┼─┼─┘
  │ │   │ │ │   │ │              │ │ │   │ │ │
┌─┼─┼───┼─┼─┼───┼─┼─┐          ┌─┼─┼─┼───┼─┼─┼─┐
│ │ │   │ │ │   │ │ │          │ │ │ │   │ │ │ │
↓ ↓ ↓   ↓ ↓ ↓   ↓ ↓ ↓          ↓ ↓ ↓ ↓   ↓ ↓ ↓ ↓

Key Features:
- Rotary Position Embeddings (RoPE)
- KV caching for efficient generation
- Optional query/key normalization
- Attention softcapping
- Causal masking

Configuration Options:
- Number of attention heads
- Head dimension size  
- Query group configuration
- RoPE scaling
- Attention bias
"""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(config.n_embd, 
                           (config.n_head + 2 * config.n_query_groups) * config.head_size,
                          bias=config.attention_bias or config.bias)
        
        self.proj = nn.Linear(config.n_head * config.head_size, config.n_embd, bias=config.bias)

        if config.norm_qk:
            self.norm_q = config.norm_class(config.head_size * config.n_head, eps=config.norm_eps)
            self.norm_k = config.norm_class(config.head_size * config.n_query_groups, eps=config.norm_eps)
        else:
            self.norm_q = self.norm_k = None 

        self.config = config
        self.block_idx = block_idx
        self.kv_cache = None
        
        # Initialize RoPE frequency computation
        if config.use_rope:
            self.freqs_complex = pre_compute_freq(
                config=config,
                context_length=config.block_size,
                device=None,
                extra_config=config.rope_scaling if hasattr(config, 'rope_scaling') else None
            )

    def initialize_kv_cache(self, batch_size: int, max_seq_len: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> None:
        """Initialize the key-value cache for this attention layer."""
        self.kv_cache = KVCache(batch_size, max_seq_len, num_kv_heads, head_dim, dtype)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                use_cache: bool = False
    ) -> torch.Tensor:
        B, T, C = x.size()  # B = batch size, T = sequence length, C = embedding dimension

        # Compute queries, keys, and values
        qkv = self.qkv(x)
        q_size = self.config.n_head * self.config.head_size
        k_size = v_size = self.config.n_query_groups * self.config.head_size
        q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

        # Normalize queries and keys if specified
        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)

        # Reshape for multi-head attention
        q = q.view(B, T, self.config.n_head, self.config.head_size).transpose(1, 2)
        k = k.view(B, T, self.config.n_query_groups, self.config.head_size).transpose(1, 2)
        v = v.view(B, T, self.config.n_query_groups, self.config.head_size).transpose(1, 2)

        # Apply RoPE to queries and keys
        if self.config.use_rope:
            if self.freqs_complex.device != q.device:
                self.freqs_complex = self.freqs_complex.to(q.device)
            q = apply_rope(q, self.freqs_complex)
            k = apply_rope(k, self.freqs_complex)

        # Handle KV caching for autoregressive generation
        if use_cache and self.kv_cache is not None:
            if self.kv_cache.size > 0:
                k_cache, v_cache = self.kv_cache.update(k, v)
                cache_size = self.kv_cache.size
                if mask is None and self.config.causal_attention:
                    mask = torch.zeros(1, 1, 1, cache_size, dtype=torch.bool, device=q.device)
                y = self.scaled_dot_product_attention(q, k_cache[:, :, :cache_size], v_cache[:, :, :cache_size], mask)
            else:
                self.kv_cache.update(k, v)
                if mask is None and self.config.causal_attention:
                    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
                    mask = mask.unsqueeze(0).unsqueeze(0)
                y = self.scaled_dot_product_attention(q, k, v, mask)
        else:
            if mask is None and self.config.causal_attention:
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
                mask = mask.unsqueeze(0).unsqueeze(0)
            y = self.scaled_dot_product_attention(q, k, v, mask)

        # Reassemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        # Output projection
        return self.proj(y)

    def scaled_dot_product_attention(self,
                                    q: torch.Tensor,
                                    k: torch.Tensor,
                                    v: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None
                                    ) -> torch.Tensor:
        
        """
Computes scaled dot-product attention with optional optimizations.

Parameters:
    q (torch.Tensor): Query tensor of shape (B, nh, T, hs)
    k (torch.Tensor): Key tensor of shape (B, nh_kv, T, hs) 
    v (torch.Tensor): Value tensor of shape (B, nh_kv, T, hs)
    mask (Optional[torch.Tensor]): Attention mask of shape (1, 1, T, T)

Returns:
    torch.Tensor: Output tensor of shape (B, nh, T, hs)

Implementation Notes:
1. Scales attention scores by 1/sqrt(head_size)
2. Handles GQA/MQA by repeating keys/values as needed
3. Supports attention logit softcapping
4. Uses PyTorch's optimized SDPA when possible
5. Applies causal masking when no explicit mask provided
"""

        scale = 1.0 / math.sqrt(self.config.head_size)

        # GQA/MQA implementation
        if self.config.n_query_groups != self.config.n_head:
            repeat_factor = self.config.n_head // self.config.n_query_groups
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        if self.config.attention_logit_softcapping is not None:
            atten_score = torch.matmul(q, k.transpose(-1, -2)) * scale
            capped_score = softcapping(atten_score, self.config.attention_logit_softcapping)
            if mask is not None:
                capped_score = capped_score.masked_fill(mask, float("-inf"))
            scores = F.softmax(capped_score, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
            y = torch.matmul(scores, v)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None if mask is None else ~mask,
                dropout_p=0.0, 
                scale=scale, 
                is_causal=mask is None and self.config.causal_attention
            )
        return y


def softcapping(x: torch.Tensor, thresh: float) -> torch.Tensor:
    """Apply softcapping to the input tensor to prevent extreme values."""
    return torch.tanh(x / thresh) * thresh


def pre_compute_freq(config, context_length=None, base=10000.0, device=None, extra_config=None):
    """
Pre-compute frequency matrix for Rotary Position Encoding (RoPE).

RoPE applies rotations to the input embeddings based on position and frequency:

For each dimension pair (2i, 2i+1):
[cos(mθ), -sin(mθ)]  ×  [x_{2i}  ]
[sin(mθ),  cos(mθ)]     [x_{2i+1}]

where:
- m is the position
- θ is the frequency for that dimension
- x_{2i}, x_{2i+1} are embedding components

Parameters:
    config (Config): Model configuration
    context_length (int, optional): Maximum sequence length
    base (float): Base for frequency computation (default: 10000.0)
    device (torch.device, optional): Device for tensor allocation
    extra_config (dict, optional): Extra configuration for dynamic NTK scaling

Returns:
    torch.Tensor: Complex tensor containing pre-computed frequencies

Features:
- Supports dynamic NTK-aware scaling
- Handles both standard and scaled RoPE
- Optimized for numerical stability
"""

    head_dim = config.n_embd // config.n_head
    context_length = context_length or config.block_size

    # Compute the inverse frequency tensor
    theta_idx = torch.arange(0, head_dim // 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (2 * theta_idx / head_dim))

    if extra_config is not None:
        orig_context_len = extra_config["original_max_seq_len"]
        factor = extra_config["factor"]
        low_freq_factor = extra_config["low_freq_factor"]
        high_freq_factor = extra_config["high_freq_factor"]

        # Compute wavelength and adjusted frequencies
        wavelen = 2 * torch.pi / inv_freq
        low_wavelength = orig_context_len / low_freq_factor
        high_wavelength = orig_context_len / high_freq_factor

        inv_freq_adj = torch.where(wavelen > low_wavelength, inv_freq / factor, inv_freq)
        smooth_factor = ((orig_context_len / wavelen) - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)

        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq
        is_medium = (wavelen <= low_wavelength) & (wavelen >= high_wavelength)
        inv_freq = torch.where(is_medium, smoothed_inv_freq, inv_freq_adj)

    # Compute frequency matrix
    positions = torch.arange(context_length, dtype=torch.float32, device=device)
    freq = torch.outer(positions, inv_freq)

    # Return complex tensor for RoPE
    return torch.polar(torch.ones_like(freq), freq)


def apply_rope(x, freqs_complex):
    """Apply Rotary Position Encoding to input tensor."""
    batch, heads, seq_len, head_dim = x.shape

    # Extract relevant sequence length from precomputed frequencies
    freqs_complex = freqs_complex[:seq_len].unsqueeze(0).unsqueeze(0)

    orig_dtype = x.dtype
    if x.dtype != torch.float32:
        x = x.float()

    # Reshape input to complex form
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_reshape)

    # Apply rotation
    x_rotate = x_complex * freqs_complex

    # Convert back to real representation
    x_rotate = torch.view_as_real(x_rotate)
    return x_rotate.reshape(*x.shape).to(orig_dtype)

class KVCache(nn.Module):
    """
Key-Value cache for efficient autoregressive generation in transformer models.

The KV cache stores previously computed key and value tensors to avoid
redundant computations during autoregressive generation. This significantly
improves inference speed for text generation.

Cache Operation:
┌────────────────┐
│ Previous Keys  │──┐
└────────────────┘  │    ┌─────────────┐
                    ├───►│ Updated      │
┌────────────────┐  │    │ Cache       │
│ New Key Batch  │──┘    │             │
└────────────────┘       │             │
                        │             │
┌────────────────┐  ┐    │             │
│ Previous Values│──┤    │             │
└────────────────┘  ├───►│             │
                    │    │             │
┌────────────────┐  │    │             │
│ New Value Batch│──┘    └─────────────┘
└────────────────┘

Key Features:
- Efficient memory management
- Automatic position tracking
- Batch-aware updates
- Device-aware initialization

Usage Pattern:
1. Initialize cache with max expected sequence length
2. For each generation step:
   a. Update cache with new keys/values
   b. Use cached values for attention computation
3. Reset cache when starting new sequence
"""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()

        # Shape: (B, H, S, D)
        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)

        # Buffers to hold cached keys and values
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)

        # Tracks positions in the sequence [0, 1, ..., max_seq_len - 1]
        self.register_buffer("cache_pos", torch.arange(0, cache_shape[2]), persistent=False)

        self.batch_size = batch_size

    def reset(self) -> None:
        """Clears the cache and resets the position to the start."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos -= self.size  # Resets to 0

    @property
    def size(self) -> int:
        """Returns the number of tokens currently stored in the cache."""
        return self.cache_pos[0].item()

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends new key and value tensors to the cache."""
        bsz, _, seq_len, _ = k_val.shape

        # Ensure incoming batch size fits the allocated cache
        if bsz > self.k_cache.shape[0]:
            raise ValueError(
                f"The current cache has a batch size of {self.k_cache.shape[0]}, "
                f"but received input with batch size {k_val.shape[0]}"
            )

        # Ensure there's enough room in the cache for the new entries
        assert (self.cache_pos[0] + seq_len) <= self.k_cache.shape[2]

        # Reference to the current key and value caches
        k_out = self.k_cache
        v_out = self.v_cache

        # Write new keys and values at the current cache position
        k_out[:bsz, :, self.cache_pos[:seq_len]] = k_val
        v_out[:bsz, :, self.cache_pos[:seq_len]] = v_val

        # Advance the cache position by the number of new tokens
        self.cache_pos.add_(seq_len)

        return k_out, v_out


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    A variant of Layer Normalization that uses RMS statistics instead of mean/variance.
    This normalization scheme is used in models like LLaMA for improved stability
    and computational efficiency.
    
    Mathematical Operation:
                        x
    y = weight * ────────────────────
                  √(mean(x²) + eps)
    
    Where:
    - x is the input tensor
    - weight is a learnable scale parameter
    - eps is a small constant for numerical stability
    
    Advantages over LayerNorm:
    1. Computationally simpler (no mean subtraction)
    2. Better numerical stability
    3. Potentially better performance on some tasks
    
    Attributes:
        weight (nn.Parameter): Learnable scale parameter
        eps (float): Small constant for numerical stability
        dim (int): Dimension along which to normalize
        add_unit_offset (bool): Whether to add 1 to the weight parameter
    """

    def __init__(self,
                size: int,
                dim: int = -1,
                eps: float = 1e-6,
                add_unit_offset: bool = False
                ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()

        # Compute mean square along the specified dimension
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)

        # Normalize the input
        x_normed = x * torch.rsqrt(norm_x + self.eps)

        # Apply learnable weight scaling
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        """Reinitialize the weight parameters."""
        torch.nn.init.ones_(self.weight)



class GptMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block used in GPT-style models.
    
    Standard MLP with GELU activation function:
    
    Architecture:
    ┌─────────┐    ┌──────┐    ┌─────────┐
    │  Linear │───►│ GELU │───►│  Linear │
    │  Layer  │    │      │    │  Layer  │
    └─────────┘    └──────┘    └─────────┘
       ×4h           f(x)          ×h
    
    where h is the hidden dimension (n_embd)
    
    Attributes:
        fc (nn.Linear): First linear transformation expanding dimensions
        proj (nn.Linear): Second linear transformation projecting back to model dimension
        config (Config): Model configuration object
    """
    
    def __init__(self, config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)
        self.proj = nn.Linear(config.mlp_hidden_size, config.n_embd, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate=self.config.gelu_approx)
        return self.proj(x)

class LLaMAMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block used in LLaMA-style models.
    
    Uses SwiGLU activation with a gating mechanism:
    
    Architecture:
                   ┌─────────┐
                   │  SiLU   │
    ┌─────────┐   └────┬────┘   ┌─────────┐
    │ Linear1 │────────┤        │         │
    └─────────┘        │   ×    │ Linear3 │
                       ├────────►│         │
    ┌─────────┐       │        │         │
    │ Linear2 │───────┘         │         │
    └─────────┘                └─────────┘
    
    The computation follows: proj(SiLU(fc1(x)) * fc2(x))
    
    Attributes:
        fc_1 (nn.Linear): First linear transformation for SiLU path
        fc_2 (nn.Linear): Second linear transformation for gating path
        proj (nn.Linear): Output projection
        config (Config): Model configuration object
    """
    
    def __init__(self, config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)
        self.proj = nn.Linear(config.mlp_hidden_size, config.n_embd, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.silu(x_fc_1) * x_fc_2
        return self.proj(x)

class GptNeoxMLP(nn.Module):
    """
    GPT-NeoX style Multi-Layer Perceptron (MLP) block.
    
    This implements the feed-forward network used in GPT-Neo and GPT-NeoX models,
    which differs from standard GPT MLP by using a configurable intermediate size
    rather than the fixed 4x expansion ratio.

    Architecture:
    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
    │               │     │               │     │               │
    │  Input Tensor │ ──► │ Intermediate  │ ──► │ Output Project│
    │   (n_embd)    │     │    Linear     │     │    Linear     │
    │               │     │ (intermediate │     │   (n_embd)    │
    └───────────────┘     │    _size)     │     │               │
                          │               │     └───────────────┘
                          └───────┬───────┘
                                  │
                                  ▼
                          ┌───────────────┐
                          │               │
                          │    GELU       │
                          │  Activation   │
                          │               │
                          └───────────────┘

    Key Characteristics:
    - Uses GELU activation (with optional approximation)
    - Configurable intermediate dimension size
    - No gating mechanism (unlike LLaMA's MLP)
    - Includes bias terms when configured

    Mathematical Formulation:
        MLP(x) = proj(gelu(fc(x)))

    Where:
        fc:      Linear projection to intermediate size
        gelu:    Gaussian Error Linear Unit activation
        proj:    Linear projection back to embedding size

    Configuration Parameters:
        n_embd:           Input/output embedding dimension
        intermediate_size: Size of the intermediate layer
        bias:             Whether to include bias terms
        gelu_approximate: Type of GELU approximation to use

    Reference:
    Based on the implementation in:
    "GPT-NeoX-20B: An Open-Source Autoregressive Language Model"
    (Black et al., 2022)
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the GPT-NeoX MLP block.

        Args:
            config (Config): Model configuration object containing:
                - n_embd: Input/output dimension
                - intermediate_size: Hidden layer dimension
                - bias: Whether to use bias in linear layers
                - gelu_approximate: Type of GELU approximation
        """
        super().__init__()
        self.fc = nn.Linear(
            config.n_embd, config.intermediate_size, bias=config.bias
        )
        self.proj = nn.Linear(
            config.intermediate_size, config.n_embd, bias=config.bias
        )
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP block.

        Args:
            x (torch.Tensor): Input tensor of shape 
                             (batch_size, seq_len, n_embd)

        Returns:
            torch.Tensor: Output tensor of same shape as input
                         (batch_size, seq_len, n_embd)

        Processing Steps:
        1. Project input to intermediate dimension
        2. Apply GELU activation
        3. Project back to original dimension
        """
        x = self.fc(x)
        x = F.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)
    

class GemmaMLP(LLaMAMLP):
    """
    Gemma-style Multi-Layer Perceptron (MLP) block.

    A variant of the LLaMA MLP that uses GELU activation instead of SiLU (Swish)
    in the gating mechanism. This provides a different non-linearity profile while
    maintaining the parameter efficiency of the gated architecture.

    Architecture:
    ┌───────────────┐     ┌───────────────┐
    │               │     │               │
    │  Input Tensor │ ──► │   Linear_1    │ ────┐
    │   (n_embd)    │     │ (intermediate │     │
    │               │     │    size)      │     │
    └───────────────┘     │               │     │
                          └───────┬───────┘     │
                                  │             │
                                  ▼             │
                          ┌───────────────┐     │
                          │               │     │
                          │     GELU      │     │
                          │  Activation   │     │
                          │               │     │
                          └───────┬───────┘     │
                                  │             │
                                  │     ┌───────┴───────┐
                                  │     │               │
                                  ├────►│   Element-    │
                                  │     │    wise       │
                                  │     │ Multiplication│
                                  │     │               │
                          ┌───────┴───────┐     │
                          │               │     │
                          │   Linear_2    │ ◄───┘
                          │ (intermediate │
                          │    size)      │
                          │               │
                          └───────┬───────┘
                                  │
                                  ▼
                          ┌───────────────┐
                          │               │
                          │ Output Project│
                          │   (n_embd)    │
                          │               │
                          └───────────────┘

    Key Characteristics:
    - Inherits from LLaMAMLP but replaces SiLU with GELU activation
    - Uses gated architecture with element-wise multiplication
    - Maintains two parallel linear transformations
    - Configurable intermediate dimension size
    - Includes bias terms when configured

    Mathematical Formulation:
        MLP(x) = proj(gelu(fc1(x)) * fc2(x))

    Where:
        fc1: First linear projection
        fc2: Second linear projection (gate)
        gelu: Gaussian Error Linear Unit activation
        *: Element-wise multiplication
        proj: Linear projection back to embedding size

    Configuration Parameters (inherited from LLaMAMLP):
        n_embd: Input/output embedding dimension
        intermediate_size: Size of intermediate layers
        bias: Whether to include bias terms
        gelu_approximate: Type of GELU approximation to use

    Reference:
    Based on the architecture described in:
    Gemma Technical Report (Google DeepMind, 2024)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Gemma MLP block.

        Args:
            x (torch.Tensor): Input tensor of shape 
                             (batch_size, seq_len, n_embd)

        Returns:
            torch.Tensor: Output tensor of same shape as input
                         (batch_size, seq_len, n_embd)

        Processing Steps:
        1. Two parallel linear projections of input
        2. GELU activation applied to first projection
        3. Element-wise multiplication of activated and gated branches
        4. Final projection back to original dimension

        Note:
        The GELU approximation type is controlled by config.gelu_approximate,
        allowing for different precision/performance tradeoffs.
        """
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x) 
    

if __name__ == '__main__':
    # lets test the model 
    from config import Config
    config = Config.from_name("gpt2-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device",device)
    model = GPT(config).to(device)
    inputs = torch.randint(0, config.vocab_size, (1, 10)).to(device)
    outputs = model(inputs)
    print(outputs.shape)  # (1, 10, vocab_size) 
    print(model)

