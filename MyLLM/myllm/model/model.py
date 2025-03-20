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

from typing import Optional , Tuple

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
            {block_idx: Block(config, block_idx) for block_idx in range(config.n_layer)}
        )

        # Final layer normalization before output
        self.ln_f = config.norm_class(config.n_embd, eps=config.norm_eps)

    def forward(self, x):
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

# Fixed CausalSelfAttention class with proper scaled_dot_product_attention method placement
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
        
        head_size = self.config.head_size
        n_query_groups = self.config.n_query_groups 
        n_head = self.config.n_head 

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
        B, T, C = x.size()
        
        # qkv: [B, T, (n_head + 2 * n_query_groups) * head_size] 
        # (B, T, C) -> (B, T, (n_head + 2 * n_query_groups) * head_size)
        qkv = self.qkv(x)
        # size of q, k, v
        q_size = n_head * head_size
        v_size = k_size = n_query_groups * head_size
        # split the qkv into q, k, v 
        # q: [B, T, n_head * head_size]
        # k: [B, T, n_query_groups * head_size]
        # v: [B, T, n_query_groups * head_size]
        q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
       
        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)
        
        # reshape the tensors 
        # (B, T, n_head * head_size) -> (B, n_head, T, head_size)
        # (B, T, n_query_groups * head_size) -> (B, n_query_groups, T, head_size)
        # (B, T, n_query_groups * head_size) -> (B, n_query_groups, T, head_size)
        q = q.view(B, T, n_head, head_size).transpose(1, 2)
        k = k.view(B, T, n_query_groups, head_size).transpose(1, 2)
        v = v.view(B, T, n_query_groups, head_size).transpose(1, 2)

        # Apply RoPE to Q and K
        if self.config.use_rope:
            # Move the freqs_complex tensor to the same device as q and k
            if self.freqs_complex.device != q.device:
                self.freqs_complex = self.freqs_complex.to(q.device)
            q = apply_rope(q, self.freqs_complex)
            k = apply_rope(k, self.freqs_complex)

        # apply the mask for causal attention
        # mask must be in the shape (1, 1, T, T)
        if mask is None and self.config.causal_attention:
            # Create a causal mask for the current sequence length
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Use scaled dot product attention 
        y = self.scaled_dot_product_attention(q, k, v, mask)

        # Re-assemble all head outputs side by side.
        y = y.reshape(B, T, head_size * n_head)

        # Output projection.
        return self.proj(y)  # (B, T, C)
    
    def scaled_dot_product_attention(self,
                                    q: torch.Tensor,  # (B, nh, T, hs)
                                    k: torch.Tensor,  # (B, nh, T, hs)
                                    v: torch.Tensor,  # (B, nh, T, hs)
                                    mask: Optional[torch.Tensor] = None  # (1, 1, T, T) or None
                                    ) -> torch.Tensor:
        """
        Computes the scaled dot-product attention.

        Args:
        - q (torch.Tensor): Query tensor of shape (B, nh, T, hs).
        - k (torch.Tensor): Key tensor of shape (B, nh, T, hs).
        - v (torch.Tensor): Value tensor of shape (B, nh, T, hs).
        - mask (Optional[torch.Tensor]): Attention mask of shape (1, 1, T, T) or None.

        Returns:
        - torch.Tensor: Output tensor of shape (B, T, nh, hs).
        """
        # Scaling factor to prevent exploding gradients
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)  

        # Check if softcapping is applied
        if self.config.attention_logit_softcapping is not None:
            # Compute raw attention scores (B, nh, T, T)
            atten_score = q @ k.transpose(-1, -2) * scale  # Matrix multiplication: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)

            # Apply softcapping to prevent extremely large values
            capped_score = softcapping(atten_score, self.config.attention_logit_softcapping)

            # Apply the mask to attention scores if provided
            if mask is not None:
                capped_score = capped_score.masked_fill(mask, float("-inf"))

            # Apply softmax over the last dimension (T) to normalize attention scores
            scores = F.softmax(capped_score, dim=-1, dtype=torch.float32).to(dtype=q.dtype)  # (B, nh, T, T)

            # Compute the attention output (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
            y = scores @ v

        else:
            # Use PyTorch's optimized attention function when no softcapping is applied
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None and self.config.causal_attention
            )  # (B, nh, T, hs)

        return y  # (B, nh, T, hs)


def softcapping(x: torch.Tensor,
                thresh: float) -> torch.Tensor:
    
    return torch.tanh(x / thresh) * thresh
    

def pre_compute_freq(config,
                    context_length=None,
                    base=10000.0,
                    device=None,
                    extra_config=None):
    """
    Pre-compute frequency matrix for Rotary Position Encoding (RoPE).
    
    This function computes the frequency tensors used for rotary embeddings, with support
    for both standard RoPE and advanced implementations with frequency scaling for
    extended context (as used in models like Llama 3).
    
    Args:
        config: Configuration object containing model parameters
        context_length: Maximum sequence length to pre-compute (defaults to config.context_length)
        base: Base value for frequency computation (default: 10000.0)
        device: Torch device to place tensors on
        extra_config: Optional dictionary for advanced RoPE configuration with keys:
            - original_max_seq_len: Original context length the model was trained with
            - factor: Scaling factor for frequency adjustment
            - low_freq_factor: Factor to determine low frequency threshold
            - high_freq_factor: Factor to determine high frequency threshold
    
    Returns:
        torch.Tensor: Complex tensor containing pre-computed rotation matrices
    """
    # Calculate dimension per attention head
    head_dim = config.n_embd // config.n_head
    
    # Use provided context_length or fall back to config
    context_length = context_length or config.context_length

    # Compute base inverse frequencies for each dimension
    # Each dimension gets a different frequency based on its position
    theta_idx = torch.arange(0, head_dim // 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (2 * theta_idx / head_dim))

    if extra_config is not None:
        # --- Advanced RoPE computation (NTK-aware, used in Llama 3) ---
        # This implementation scales frequencies differently based on their wavelength,
        # allowing for better extrapolation to longer sequences
        
        orig_context_len = extra_config["original_max_seq_len"]
        factor = extra_config["factor"]  # Scaling factor (e.g., 2.0 for doubling context)
        low_freq_factor = extra_config["low_freq_factor"]  # Threshold for low frequencies
        high_freq_factor = extra_config["high_freq_factor"]  # Threshold for high frequencies

        # Compute wavelength thresholds
        # Wavelength = 2π/frequency, so high frequency = low wavelength and vice versa
        low_wavelength = orig_context_len / low_freq_factor  # Threshold for scaling
        high_wavelength = orig_context_len / high_freq_factor
        
        # Calculate wavelength for each frequency component
        wavelen = 2 * torch.pi / inv_freq
        
        # For low frequencies (high wavelengths > low_wavelength):
        # Scale these frequencies by dividing by factor (slowing them down)
        inv_freq_adj = torch.where(wavelen > low_wavelength,
                                  inv_freq / factor,
                                  inv_freq)
        
        # For medium frequencies - apply a smooth transition between scaled and unscaled
        # This prevents abrupt changes in the attention pattern
        smooth_factor = ((orig_context_len / wavelen) - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)  # Ensure factor is between 0 and 1
        
        # Blend between scaled and unscaled frequencies based on smooth_factor
        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq
        
        # Apply smoothed frequencies for medium-range wavelengths
        is_medium = (wavelen <= low_wavelength) & (wavelen >= high_wavelength)
        inv_freq = torch.where(is_medium, smoothed_inv_freq, inv_freq_adj)
    
    # Compute position-frequency outer product:
    # For each position and frequency, compute position × frequency
    positions = torch.arange(context_length, dtype=torch.float32, device=device)
    freq = torch.outer(positions, inv_freq)
    
    # Convert to complex rotation factors (e^(i·θ) = cos(θ) + i·sin(θ))
    return torch.polar(torch.ones_like(freq), freq)


def apply_rope(x, freqs_complex):
    """
    Apply Rotary Position Encoding to input tensor.
    
    This function applies pre-computed rotary embeddings to the input tensor.
    It works by interpreting pairs of features as complex numbers and multiplying
    them by the pre-computed complex rotation factors.
    
    Args:
        x: Input tensor of shape [batch, heads, seq_len, head_dim]
        freqs_complex: Pre-computed complex rotation matrix from pre_compute_freq
    
    Returns:
        torch.Tensor: Tensor with rotary position encoding applied
    """
    # Get the actual sequence length from the input tensor
    _, _, seq_len, _ = x.shape
    
    # Truncate frequency tensor to the actual sequence length
    freqs_complex = freqs_complex[:seq_len]
    
    # Add batch and head dimensions to match input tensor
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    
    # Save original dtype for restoration later
    orig_dtype = x.dtype
    
    # Convert to float32 if needed for complex number operations
    if x.dtype != torch.float32:
        x = x.float()
    
    # Reshape to interpret adjacent dimensions as real and imaginary components
    # Last dimension becomes [dim/2, 2] where each pair is (real, imag)
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    
    # Convert to complex numbers
    x_complex = torch.view_as_complex(x_reshape)
    
    # Apply rotation by multiplying with complex rotation factors
    # This is equivalent to:
    # [cos(θ) + i·sin(θ)] × [a + i·b] = [a·cos(θ) - b·sin(θ)] + i·[a·sin(θ) + b·cos(θ)]
    x_rotate = x_complex * freqs_complex
    
    # Convert back to real representation
    x_rotate = torch.view_as_real(x_rotate)
    
    # Restore original shape and dtype
    return x_rotate.reshape(*x.shape).to(orig_dtype)


class GPTMLP(nn.Module):
    pass 


class LlamaMLP(nn.Module):
    pass 



class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight) 


class KV_Cache(nn.Module):
    pass 



    
    
        
      

    

