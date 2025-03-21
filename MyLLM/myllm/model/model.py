# Decoder-Only Transformer Model Implementation (GPT-style)
#
# This module provides a flexible and scalable implementation of a decoder-only
# transformer model, similar to the GPT architecture. The implementation is designed
# to support various configurations and architectural variations. It is extensible 
# for different transformer-based models and can be adapted for custom modifications 
# like GPT-2, GPT-Neo, or LLaMA.


    
# Importing necessary libraries for model implementation
import torch  
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Tuple

# Import the configuration class from the config module
from config import Config 

class GPT(nn.Module):
    """
    A GPT-like transformer model, designed as a decoder-only architecture.
    This model supports various configurations and can be easily extended for different GPT-like models.
    
    Attributes:
    ----------
    config: Config
        Configuration object that contains hyperparameters for the model such as 
        vocabulary size, embedding dimensions, number of layers, etc.
    lm_head: nn.Linear
        Linear layer that maps the embedding space to the vocabulary space for output logits.
    wte: nn.Embedding
        Token embedding layer that converts token IDs into dense vector representations.
    wpe: nn.Embedding
        Position embedding layer used to encode the position of tokens in the sequence (specific to GPT-2).
    transformer: nn.ModuleDict
        A dictionary of transformer blocks (decoder layers), each block is a separate module.
    ln_f: nn.Module
        Final layer normalization applied after all transformer blocks.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the GPT model.

        Parameters:
        -----------
        config: Config
            Configuration object containing hyperparameters for the model, such as 
            vocabulary size, embedding dimensions, number of layers, etc.
        """
        super().__init__()
        self.config = config

        # Validate configuration
        if not hasattr(config, 'padded_vocab_size'):
            raise ValueError("Config must specify 'padded_vocab_size'.")
        if not hasattr(config, 'n_embd'):
            raise ValueError("Config must specify 'n_embd'.")

        # Linear layer to map from embedding size to vocabulary size for output logits (language modeling)
        self.lm_head = nn.Linear(
            config.padded_vocab_size, config.n_embd, bias=config.lm_head_bias
        )

        # Embedding layer for token IDs (converts tokens to embeddings of size n_embd)
        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        
        # GPT-2 specific: position embeddings (maps sequence positions to embeddings of size n_embd)
        if config.name == "gpt2":
            self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # Transformer blocks (decoder layers)
        # A ModuleDict to store multiple transformer blocks, each one being an individual layer
        self.transformer = nn.ModuleDict(
            {f"block_{block_idx}": Block(config, block_idx) for block_idx in range(config.n_layer)}
        )

        # Final layer normalization before output
        self.ln_f = config.norm_class(config.n_embd, eps=config.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GPT model.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch_size, seq_len), containing token indices.

        Returns:
        --------
        logits: torch.Tensor
            Logits of shape (batch_size, seq_len, vocab_size) representing unnormalized 
            probabilities for each token in the vocabulary.
        """
        B, T = x.size()  # B = batch size, T = sequence length

        # Check if input sequence length is within the allowed block size
        if T > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {T} tokens, block size is only {self.config.block_size}. "
                "This is likely because the input text exceeds the supported context length of this model."
            )

        # Add token embeddings: Convert token IDs into embeddings of size n_embd
        token_embeddings = self.wte(x)

        # If the model is GPT-2, we also need to add position embeddings
        if self.config.name == "gpt2":
            # Generate position indices for each token (0, 1, 2, ..., seq_len-1)
            pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
            # Get position embeddings corresponding to each token's position
            position_embeddings = self.wpe(pos)
            # Add the token embeddings with the position embeddings
            x = token_embeddings + position_embeddings
        else:
            # For non-GPT-2 models, only token embeddings are used
            x = token_embeddings

        # Now, pass the tensor through each transformer block (decoder layers)
        # The output of each block will be passed into the next block
        for block in self.transformer.values():
            x = block(x)

        # After passing through all transformer blocks, apply final layer normalization
        x = self.ln_f(x)

        # The final output logits are produced by applying the lm_head (Linear layer)
        logits = self.lm_head(x)

        return logits


class Block(nn.Module):
    """
    A single transformer block (decoder layer), composed of multi-head self-attention and a position-wise feed-forward network.
    
    Attributes:
    ----------
    norm1: nn.Module
        The normalization layer applied before self-attention.
    norm2: nn.Module, optional
        The normalization layer applied before the feed-forward network.
    attn: CausalSelfAttention
        The causal self-attention mechanism that performs attention on the input sequence.
    post_attention_norm: nn.Module
        Post-attention normalization, applied after the self-attention mechanism.
    post_mlp_norm: nn.Module
        Post-MLP normalization, applied after the feed-forward network.
    mlp: nn.Module
        The multi-layer perceptron (MLP) component of the transformer block.
    """
    
    def __init__(self, config: Config, block_idx: int) -> None:
        """
        Initialize a single transformer block (decoder layer).

        Parameters:
        -----------
        config: Config
            Configuration object containing hyperparameters for the block, such as 
            embedding dimensions, number of attention heads, etc.
        block_idx: int
            The index of the block in the model (used for layer-specific configurations).
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───────►    norm_2
           │  ↓                     │  ↓                   ↓
           │  attn                  │  attn                MLP
           │  ↓                     │  ↓                   ↓
           |  post_attn_norm        |  post_attn_norm      post_mlp_norm
           |  ↓                     |  ↓                   ↓
        ┌─ └► +                     └► + ◄─────────────────┘
        |     ↓
        │     norm_2
        │     ↓
        │     MLP
        │     ↓
        |     post_mlp_norm
        |     ↓
        └───► +
        """
        # Apply the first normalization
        x_normed = self.norm1(x)

        # Apply self-attention
        attn_out = self.attn(x_normed)

        # Apply post-attention normalization
        attn_out = self.post_attention_norm(attn_out)

        if self.config.parallel_residual:
            # Shared norm case: use x_normed, otherwise apply norm2
            x_normed = self.norm2(x) if self.norm2 is not None else x
            
            # Apply MLP
            mlp_out = self.mlp(x_normed)

            # Sum attention and MLP outputs in parallel
            x = attn_out + mlp_out + x  
        else:
            # Standard residual: add attention output first
            x = x + attn_out

            # Apply second norm if necessary
            x_normed = self.norm2(x) if self.norm2 is not None else x

            # Apply MLP
            x = self.mlp(x_normed)

        # Apply post-MLP normalization
        return self.post_mlp_norm(x)


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention layer for the GPT-like model. This layer performs masked self-attention 
    where each token can only attend to previous tokens in the sequence (causal attention).
    
    Attributes:
    ----------
    config: Config
        Configuration object containing hyperparameters for the attention mechanism.
    block_idx: int
        The index of the block, used for specific layer configurations.
    """
    def __init__(self, config: Config, block_idx: int) -> None:
        """
        Initialize the Causal Self-Attention layer.

        Parameters:
        -----------
        config: Config
            Configuration object containing hyperparameters such as the number of attention heads and the embedding size.
        block_idx: int
            The index of the block for layer-specific configurations.
        """
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
        
        # Initialize RoPE frequency computation
        if config.use_rope:
            # Pre-compute RoPE frequencies during initialization
            self.freqs_complex = pre_compute_freq(
                config=config,
                context_length=config.block_size,
                device=None,  # Will be moved to appropriate device during forward pass
                extra_config=config.rope_scaling if hasattr(config, 'rope_scaling') else None
            )

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the causal self-attention layer.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch_size, seq_len, n_embd).
        mask: Optional[torch.Tensor]
            Optional attention mask of shape (1, 1, seq_len, seq_len).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, n_embd).
        """

         # to use multi-head attention (MHA), set this to `n_head` (default)
        # to use multi-query attention (MQA), set this to 1
        # to use grouped-query attention (GQA), set this to a value in between
        # Example with `n_head=4`
        # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        #   │    │    │    │         │        │                 │
        # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
        #         MHA                    GQA                   MQA
        #   n_query_groups=4       n_query_groups=2      n_query_groups=1
        #
        # credit https://arxiv.org/pdf/2305.13245.pdf
        
        # Notation : 
        # - B          | batch size
        # - T          | time-step (sequence length)
        # - C          | embedding dimension
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

        # Create causal mask if not provided
        if mask is None and self.config.causal_attention:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Compute scaled dot-product attention
        y = self.scaled_dot_product_attention(q, k, v, mask)

        # Reassemble all head outputs
        y = y.transpose(1, 2).reshape(B, T, C)

        # Output projection
        return self.proj(y)

    def scaled_dot_product_attention(self,
                                    q: torch.Tensor,
                                    k: torch.Tensor,
                                    v: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None
                                    ) -> torch.Tensor:
        """
        Computes the scaled dot-product attention.

        Args:
        - q (torch.Tensor): Query tensor of shape (B, nh, T, hs).
        - k (torch.Tensor): Key tensor of shape (B, nh, T, hs).
        - v (torch.Tensor): Value tensor of shape (B, nh, T, hs).
        - mask (Optional[torch.Tensor]): Attention mask of shape (1, 1, T, T) or None.

        Returns:
        - torch.Tensor: Output tensor of shape (B, nh, T, hs).
        """
        scale = 1.0 / math.sqrt(self.config.head_size)

        if self.config.attention_logit_softcapping is not None:
            atten_score = q @ k.transpose(-1, -2) * scale
            capped_score = softcapping(atten_score, self.config.attention_logit_softcapping)
            if mask is not None:
                capped_score = capped_score.masked_fill(mask, float("-inf"))
            scores = F.softmax(capped_score, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
            y = scores @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None and self.config.causal_attention
            )
        return y


def softcapping(x: torch.Tensor, thresh: float) -> torch.Tensor:
    """
    Apply softcapping to the input tensor to prevent extreme values.

    Args:
    - x (torch.Tensor): Input tensor.
    - thresh (float): Threshold for softcapping.

    Returns:
    - torch.Tensor: Softcapped tensor.
    """
    return torch.tanh(x / thresh) * thresh


def pre_compute_freq(config, context_length=None, base=10000.0, device=None, extra_config=None):
    """
    Pre-compute frequency matrix for Rotary Position Encoding (RoPE).
    
    Args:
        config: Configuration object containing model parameters.
        context_length: Maximum sequence length to pre-compute (defaults to config.context_length).
        base: Base value for frequency computation (default: 10000.0).
        device: Torch device to place tensors on.
        extra_config: Optional dictionary for advanced RoPE configuration.

    Returns:
        torch.Tensor: Complex tensor containing pre-computed rotation matrices.
    """
    head_dim = config.n_embd // config.n_head
    context_length = context_length or config.block_size

    theta_idx = torch.arange(0, head_dim // 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (2 * theta_idx / head_dim))

    if extra_config is not None:
        orig_context_len = extra_config["original_max_seq_len"]
        factor = extra_config["factor"]
        low_freq_factor = extra_config["low_freq_factor"]
        high_freq_factor = extra_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq
        low_wavelength = orig_context_len / low_freq_factor
        high_wavelength = orig_context_len / high_freq_factor

        inv_freq_adj = torch.where(wavelen > low_wavelength, inv_freq / factor, inv_freq)
        smooth_factor = ((orig_context_len / wavelen) - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)
        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq
        is_medium = (wavelen <= low_wavelength) & (wavelen >= high_wavelength)
        inv_freq = torch.where(is_medium, smoothed_inv_freq, inv_freq_adj)
    
    positions = torch.arange(context_length, dtype=torch.float32, device=device)
    freq = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freq), freq)


def apply_rope(x, freqs_complex):
    """
    Apply Rotary Position Encoding to input tensor.
    
    Args:
        x: Input tensor of shape [batch, heads, seq_len, head_dim].
        freqs_complex: Pre-computed complex rotation matrix from pre_compute_freq.

    Returns:
        torch.Tensor: Tensor with rotary position encoding applied.
    """
    _, _, seq_len, _ = x.shape
    freqs_complex = freqs_complex[:seq_len].unsqueeze(0).unsqueeze(0)
    orig_dtype = x.dtype
    if x.dtype != torch.float32:
        x = x.float()
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_reshape)
    x_rotate = x_complex * freqs_complex
    x_rotate = torch.view_as_real(x_rotate)
    return x_rotate.reshape(*x.shape).to(orig_dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


class GPTMLP(nn.Module):
    pass 


class LlamaMLP(nn.Module):
    pass  

class KVCache(nn.Module):
    """
    Buffers `k`, `v` have shape
    `(batch_size, n_query_groups, max_seq_length, head_size)`.
    """
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Writes new values `k` and `v` into the cache at the positions specified
        by `input_pos` along the sequence dimension (`max_seq_length`). The batch
        size of `k` and `v` (`bs`) must be smaller or equal to `KVCache` batch
        size. Returns the full buffers, adjusted to the batch size `bs`.

        Args:
            input_pos: Position index, `(bs, T)` or `(T,)`
            k: New values, `(bs, n_query_groups, T, head_size)`
            v: New values, `(bs, n_query_groups, T, head_size)`

        Returns:
            k_full, v_full, `(bs, n_query_groups, max_seq_length, head_size)`

        """
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        bs = k.size(0)
        k = batched_index_copy_(self.k[:bs, ...], -2, input_pos, k)
        v = batched_index_copy_(self.v[:bs, ...], -2, input_pos, v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)
