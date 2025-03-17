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

from typing import Optional

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
        self.qkv = nn.Linear(Config.n_embd, 
                             (Config.n_head + 2 * Config.n_query_groups) * Config.head_size,
                            bias= Config.attention_bias or Config.bias)
        
        self.proj = nn.Linear(Config.n_head * Config.head_size, Config.n_embd, bias=Config.bias)

        if Config.norm_qk:
            self.norm_q = config.norm_class(Config.head_size * Config.n_head, eps=config.norm_eps)
            self.norm_k = config.norm_class(Config.head_size * Config.n_query_groups, eps=config.norm_eps)
        else:
            self.norm_q = self.norm_k = None 

        self.config = config
        self.block_idx = block_idx

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
        
        head_size = self.config.head_size
        n_query_groups = self.config.n_query_groups 
        n_head = self.config.n_head 

        # Notation : 
        # - B          | batch size
        # - T          | time-step (sequence length)
        # - C          | embedding dimension
        B , T , C = x.size()
        # qkv: [B, T, (n_head + 2 * n_query_groups) * head_size] 
        # (B, T, C) -> (B, T, (n_head + 2 * n_query_groups) * head_size)
        qkv = self.qkv(x)
        # soze of q, k, v
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
        q = q.view(B, n_head, T, head_size)
        k = k.view(B, n_query_groups, T, head_size)
        v = v.view(B, n_query_groups, T, head_size)

        # transpose the tensors 
        # (B, n_head, T, head_size) -> (B, T, n_head, head_size)
        # (B, n_query_groups, T, head_size) -> (B, T, n_query_groups, head_size)
        # (B, n_query_groups, T, head_size) -> (B, T, n_query_groups, head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        
      

    

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