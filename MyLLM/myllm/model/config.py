"""
Configuration Management for Transformer-based Models

This module defines a `Config` class to manage the configuration settings of various transformer-based models, including models like GPT-2 and LLaMA. It provides functionality for:

1. Defining core parameters for the model architecture, such as:
    - `block_size`: Sequence length for input data.
    - `vocab_size`: The number of tokens in the vocabulary.
    - `n_layer`: Number of transformer layers.
    - `n_head`: Number of attention heads.
    - `n_embd`: Embedding dimensionality.
    - And other model-specific parameters.

2. Architecture variations for flexibility, such as different normalization layers and activation functions.

3. Model-specific parameters (e.g., for LLaMA models, including rotary embeddings and parallel residual connections).

4. Hyperparameters such as dropout rate, learning rate, and Adam optimizer settings.

5. The ability to save and load configurations from JSON files, facilitating easy configuration management.

6. Configuration validation checks (e.g., ensuring that `n_embd` is divisible by `n_head`).

7. The ability to update specific configuration parameters dynamically.

8. Retrieval of only the trainable parameters from the configuration.

9. The ability to manage multiple configurations (e.g., for different model architectures like GPT-2 and LLaMA) with an easy-to-use registry.

The module is designed to be used for managing large-scale transformer models with customizable settings, allowing easy integration with training pipelines.

Example usage:
- Create a `Config` instance for a specific model using `Config.from_name()`.
- Validate the configuration using `.validate()`.
- Save and load configurations from disk using `.save()` and `.load()`.
"""

# Import statements
from dataclasses import dataclass, field
from typing import Optional, Any, Dict , Literal
import json

@dataclass
class Config:
    # Core parameters
    name: str = ""  # Name of the model configuration
    block_size: int = 1024  # Size of each block (sequence length)
    vocab_size: int = 50257  # Number of tokens in the vocabulary
    padded_vocab_size: Optional[int] = None  # Padded vocabulary size (if applicable)
    n_layer: int = 12  # Number of transformer layers
    n_head: int = 12  # Number of attention heads
    n_embd: int = 768  # Dimensionality of embeddings
    eps: float = 1e-5  # Small epsilon for numerical stability
    head_size: Optional[int] = None


    # Architecture variations
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm" # Type of normalization layer used (LayerNorm or RMSNorm)
    activation: str = "gelu"  # Activation function (gelu, relu, etc.)
    mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = "GptNeoxMLP"
    scale_embeddings: bool = False  # Whether to scale embeddings by sqrt(d_model)
    mlp_ratio: float = 4.0  # Ratio of hidden dimension to embedding dimension in the MLP
    lm_head_bias: bool = False
    attention_bias : bool = False  # Whether to use attention bias
    bias : bool = False


    rotary_percentage: float = 0.0  # Percentage for rotary embeddings (specific to LLaMA models)
    parallel_residual: bool = False  # Whether to use parallel residual connections (specific to LLaMA)
    shared_attention_norm : bool = False  # Whether to use shared attention norm (specific to LLaMA)
    norm_eps: float = 1e-5  # Small epsilon for normalization (specific to LLaMA)
    n_query_groups: int = 32  # Number of query groups (specific to LLaMA)
    norm_qk: bool = False  # Whether to use normalized queries and keys
    use_rope : bool = False # Whether to use rope embeddings
    rope_base: int = 10000

    attention_scores_scalar : Optional[int] = None 
    softcapping_threshold : Optional[float] = None 
    attention_logit_softcapping = Optional[float] = None 


    # Hyperparameter
    dropout: float = 0.1  # Dropout rate for regularization
    bias: bool = False  # Whether to use bias terms in layers
    learning_rate: float = 3e-4  # Learning rate for training
    weight_decay: float = 0.1  # Weight decay for regularization
    beta1: float = 0.9  # First momentum term for Adam optimizer
    beta2: float = 0.999  # Second momentum term for Adam optimizer

    # Extra parameters for flexibility
    extra_params: Dict[str, Any] = field(default_factory=dict)  # To store any extra parameters

    def __post_init__(self):
        # Ensure that padded_vocab_size is set if not provided
        if self.padded_vocab_size is None:
            self.padded_vocab_size = self.vocab_size
        
        # Validate the configuration parameters after initialization
        self.validate()

    def __repr__(self):
        # Return a string representation of the config with key-value pairs
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"Config({params})"

    def save(self, file_path: str):
        """ Save the configuration to a JSON file. """
        with open(file_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load(cls, file_path: str):
        """ Load the configuration from a JSON file. """
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def update(self, **kwargs):
        """ Update the configuration with new key-value pairs. """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Invalid config key '{key}', skipping update.")

    def get_trainable_params(self):
        """ Get a dictionary of trainable parameters (those that are int, float, or bool). """
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, bool))}

    def validate(self):
        """ Validate the configuration parameters. """
        # Ensure that n_embd is divisible by n_head for correct attention behavior
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        # Ensure that block_size is positive
        assert self.block_size > 0, "block_size must be positive"
        # Ensure that mlp_ratio is positive for proper scaling of the MLP layers
        assert self.mlp_ratio > 0, "mlp_ratio must be positive"
        print("✅ All checks passed.")

    @classmethod
    def available_configs(cls):
        """ Return the list of available configurations by accessing the global configuration registry. """
        return list(name_to_config.keys())

    @classmethod
    def from_name(cls, name: str):
        """ Create a Config instance from a configuration name. """
        if name not in name_to_config:
            raise ValueError(f"Config with name {name} not found.")
        return cls(**name_to_config[name])


#  google docs string this 


# Configuration registry (you can add more configurations here as needed)
configs = [
    dict(name="gpt2-small", block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, norm_class_name="LayerNorm", mlp_class_name="GptMLP", activation="gelu", scale_embeddings=True),
    dict(name="gpt2-medium", block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024, norm_class_name="LayerNorm", mlp_class_name="GptMLP", activation="gelu", scale_embeddings=True),
    dict(name="gpt2-large", block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280, norm_class_name="LayerNorm", mlp_class_name="GptMLP", activation="gelu", scale_embeddings=True),
    dict(name="gpt2-xl", block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600, norm_class_name="LayerNorm", mlp_class_name="GptMLP", activation="gelu", scale_embeddings=True),
    dict(name="llama2-7b", block_size=4096, vocab_size=32000, n_layer=32, n_head=32, n_embd=4096, norm_class_name="RMSNorm", mlp_class_name="LLaMAMLP", rotary_percentage=1.0, parallel_residual=True, norm_eps=1e-5),
    dict(name="llama2-13b", block_size=4096, vocab_size=32000, n_layer=40, n_head=40, n_embd=5120, norm_class_name="RMSNorm", mlp_class_name="LLaMAMLP", rotary_percentage=1.0, parallel_residual=True, norm_eps=1e-5),
    dict(name="llama3-8b", block_size=8192, vocab_size=128256, n_layer=32, n_head=32, n_embd=4096, norm_class_name="RMSNorm", mlp_class_name="LLaMAMLP", rotary_percentage=1.0, parallel_residual=True, norm_eps=1e-5),
    dict(name="llama3-70b", block_size=8192, vocab_size=128256, n_layer=80, n_head=64, n_embd=8192, norm_class_name="RMSNorm", mlp_class_name="LLaMAMLP", rotary_percentage=1.0, parallel_residual=True, norm_eps=1e-5)
]

# Create a mapping of model names to configurations
name_to_config = {config["name"]: config for config in configs}
''''''

