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
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias
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
        y = y.transpose(1, 2).contiguous().view(B, T, -1)  # Fix: Use -1 to infer the correct dimension

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

        if self.config.n_query_groups != self.config.n_head:
        # Repeat keys and values to match the number of heads
            repeat_factor = self.config.n_head // self.config.n_query_groups
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

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


import torch

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
        torch.Tensor: Complex tensor of shape [context_length, head_dim // 2].
    """
    head_dim = config.n_embd // config.n_head  # Scalar
    context_length = context_length or config.block_size  # Scalar

    # Compute the inverse frequency tensor
    theta_idx = torch.arange(0, head_dim // 2, dtype=torch.float32, device=device)  # Shape: [head_dim // 2]
    inv_freq = 1.0 / (base ** (2 * theta_idx / head_dim))  # Shape: [head_dim // 2]

    if extra_config is not None:
        orig_context_len = extra_config["original_max_seq_len"]
        factor = extra_config["factor"]
        low_freq_factor = extra_config["low_freq_factor"]
        high_freq_factor = extra_config["high_freq_factor"]

        # Compute wavelength and adjusted frequencies
        wavelen = 2 * torch.pi / inv_freq  # Shape: [head_dim // 2]
        low_wavelength = orig_context_len / low_freq_factor  # Scalar
        high_wavelength = orig_context_len / high_freq_factor  # Scalar

        inv_freq_adj = torch.where(wavelen > low_wavelength, inv_freq / factor, inv_freq)  # Shape: [head_dim // 2]
        smooth_factor = ((orig_context_len / wavelen) - low_freq_factor) / (high_freq_factor - low_freq_factor)  # Shape: [head_dim // 2]
        smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)  # Shape: [head_dim // 2]

        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq  # Shape: [head_dim // 2]
        is_medium = (wavelen <= low_wavelength) & (wavelen >= high_wavelength)  # Shape: [head_dim // 2]
        inv_freq = torch.where(is_medium, smoothed_inv_freq, inv_freq_adj)  # Shape: [head_dim // 2]

    # Compute frequency matrix
    positions = torch.arange(context_length, dtype=torch.float32, device=device)  # Shape: [context_length]
    freq = torch.outer(positions, inv_freq)  # Shape: [context_length, head_dim // 2]

    # Return complex tensor for RoPE
    return torch.polar(torch.ones_like(freq), freq)  # Shape: [context_length, head_dim // 2]


def apply_rope(x, freqs_complex):
    """
    Apply Rotary Position Encoding to input tensor.

    Args:
        x: Input tensor of shape [batch, heads, seq_len, head_dim].
        freqs_complex: Pre-computed complex rotation matrix from pre_compute_freq.

    Returns:
        torch.Tensor: Tensor with rotary position encoding applied, same shape as x.
    """
    batch, heads, seq_len, head_dim = x.shape  # [batch, heads, seq_len, head_dim]

    # Extract relevant sequence length from precomputed frequencies
    freqs_complex = freqs_complex[:seq_len].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_len, head_dim // 2]

    orig_dtype = x.dtype
    if x.dtype != torch.float32:
        x = x.float()

    # Reshape input to complex form
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)  # Shape: [batch, heads, seq_len, head_dim // 2, 2]
    x_complex = torch.view_as_complex(x_reshape)  # Shape: [batch, heads, seq_len, head_dim // 2]

    # Apply rotation
    x_rotate = x_complex * freqs_complex  # Shape: [batch, heads, seq_len, head_dim // 2]

    # Convert back to real representation
    x_rotate = torch.view_as_real(x_rotate)  # Shape: [batch, heads, seq_len, head_dim // 2, 2]
    return x_rotate.reshape(*x.shape).to(orig_dtype)  # Shape: [batch, heads, seq_len, head_dim]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self,
                size: int,
                dim: int = -1,
                eps: float = 1e-6,
                add_unit_offset: bool = False
                ) -> None:
        """
        Args:
            size (int): The number of features in the input tensor (last dimension size).
            dim (int): The dimension along which to compute the RMS normalization (default: -1).
            eps (float): A small constant for numerical stability (default: 1e-6).
            add_unit_offset (bool): Whether to add a unit offset to the weight parameter (default: False).
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))  # Shape: [size]
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Root Mean Square Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, ..., size].

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        dtype = x.dtype
        x = x.float()  # Ensure computation is in float32 for numerical stability

        # Compute mean square along the specified dimension
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)  # Shape: [batch_size, ..., 1]

        # Normalize the input
        x_normed = x * torch.rsqrt(norm_x + self.eps)  # Shape: [batch_size, ..., size]

        # Apply learnable weight scaling
        weight = (1 + self.weight) if self.add_unit_offset else self.weight  # Shape: [size]
        return (x_normed * weight.float()).to(dtype=dtype)  # Shape: [batch_size, ..., size]

    def reset_parameters(self) -> None:
        """Reinitialize the weight parameters."""
        torch.nn.init.ones_(self.weight)  # Shape: [size]



class GptMLP(nn.Module):
    """MLP block used in GPT-like models."""
    
    def __init__(self, config) -> None:
        """
        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)  # Shape: [batch_size, seq_len, mlp_hidden_size]
        self.proj = nn.Linear(config.mlp_hidden_size, config.n_embd, bias=config.bias)  # Shape: [batch_size, seq_len, n_embd]
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, n_embd].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, n_embd].
        """
        x = self.fc(x)  # Shape: [batch_size, seq_len, mlp_hidden_size]
        x = F.gelu(x, approximate=self.config.gelu_approx)  # GELU activation (same shape)
        return self.proj(x)  # Shape: [batch_size, seq_len, n_embd]


class LLaMAMLP(nn.Module):
    """MLP block used in LLaMA-like models with Gated Activation Units (GAUs)."""
    
    def __init__(self, config) -> None:
        """
        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)  # Shape: [batch_size, seq_len, mlp_hidden_size]
        self.fc_2 = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)  # Shape: [batch_size, seq_len, mlp_hidden_size]
        self.proj = nn.Linear(config.mlp_hidden_size, config.n_embd, bias=config.bias)  # Shape: [batch_size, seq_len, n_embd]
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, n_embd].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, n_embd].
        """
        x_fc_1 = self.fc_1(x)  # Shape: [batch_size, seq_len, mlp_hidden_size]
        x_fc_2 = self.fc_2(x)  # Shape: [batch_size, seq_len, mlp_hidden_size]
        x = F.silu(x_fc_1) * x_fc_2  # Shape: [batch_size, seq_len, mlp_hidden_size] (Element-wise multiplication)
        return self.proj(x)  # Shape: [batch_size, seq_len, n_embd]



class KVCache(nn.Module):
    """
    A key-value cache module for transformer models to enable efficient autoregressive decoding.

    Stores past key and value tensors and provides an update method to append new entries.

    Args:
        batch_size (int): Expected batch size during inference.
        max_seq_len (int): Maximum sequence length (capacity of the cache).
        num_kv_heads (int): Number of attention heads for keys and values.
        head_dim (int): Dimensionality of each attention head.
        dtype (torch.dtype): Data type for the cache tensors.
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
        # B = batch_size, H = num_kv_heads, S = max_seq_len, D = head_dim
        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)

        # Buffers to hold cached keys and values
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)

        # Tracks positions in the sequence [0, 1, ..., max_seq_len - 1]
        # Shape: (max_seq_len,)
        self.register_buffer("cache_pos", torch.arange(0, cache_shape[2]), persistent=False)

        self.batch_size = batch_size

    def reset(self) -> None:
        """
        Clears the cache and resets the position to the start.
        """
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos -= self.size  # Resets to 0

    @property
    def size(self) -> int:
        """
        Returns:
            int: The number of tokens currently stored in the cache.
        """
        return self.cache_pos[0].item()

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new key and value tensors to the cache.

        Args:
            k_val (torch.Tensor): Key tensor of shape (B, H, S_new, D)
            v_val (torch.Tensor): Value tensor of shape (B, H, S_new, D)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                Updated caches:
                    - k_cache: Tensor of shape (B, H, max_seq_len, D)
                    - v_cache: Tensor of shape (B, H, max_seq_len, D)

        Raises:
            ValueError: If the new batch size exceeds the initialized batch size.
            AssertionError: If adding the new sequence would exceed max_seq_len.
        """
        bsz, _, seq_len, _ = k_val.shape  # (B, H, S_new, D)

        # Ensure incoming batch size fits the allocated cache
        if bsz > self.k_cache.shape[0]:
            raise ValueError(
                f"The current cache has a batch size of {self.k_cache.shape[0]}, "
                f"but received input with batch size {k_val.shape[0]}"
            )

        # Ensure there's enough room in the cache for the new entries
        assert (self.cache_pos[0] + seq_len) <= self.k_cache.shape[2]

        # Reference to the current key and value caches
        k_out = self.k_cache  # Shape: (B, H, max_seq_len, D)
        v_out = self.v_cache  # Shape: (B, H, max_seq_len, D)

        # Write new keys and values at the current cache position
        # Write into: [:, :, current_pos:current_pos+seq_len]
        k_out[:, :, self.cache_pos[:seq_len]] = k_val
        v_out[:, :, self.cache_pos[:seq_len]] = v_val

        # Advance the cache position by the number of new tokens
        self.cache_pos.add_(seq_len)

        return k_out, v_out


def batched_index_copy_(t, dim, idx):
    pass 



# test with GPT2 small 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of your configuration
config = Config.from_name("gpt2-small")  # Or any other configuration name

config.n_query_groups = config.n_head = 12 

# Instantiate the GPT model with your config
model = GPT(config).to(device)

print(model)

# Prepare input data
batch_size = 4
sequence_length = 128
input_tokens = torch.randint(0, config.vocab_size, (batch_size, sequence_length)).to(device)

# Forward pass
logits = model(input_tokens) 
print(logits.shape)  # Output: torch.Size([4, 128, vocab_size]) 


# TODO:
# 1. Add tests for the new features and configurations.
# 2. Implement the KVCache class and its methods.
# 3. Inject the KVCache into the model's forward pass.
# finlize the test_model.py
# 