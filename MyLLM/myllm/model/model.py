# Decoder-Only Transformer Model Implementation (GPT-style)

# This module provides a flexible and scalable implementation of a decoder-only
# transformer model, similar to the GPT architecture. The implementation is designed
# to support various configurations and architectural variations.

# Import torch library for model implementation
import torch  
import torch.nn as nn
import torch.nn.functional as F

# Import the configuration 
from config import Config 

class GPT(nn.Module):
    """
    A GPT-like transformer model, designed as a decoder-only architecture.
    This model supports various configurations and can be easily extended for different GPT-like models.
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
        
        # x is expected to have the shape (batch_size, seq_len), so we unpack the size
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
        # The logits will be of shape (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits


class Block(nn.Module):
    def __init__(self, config:Config)->None:
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    
