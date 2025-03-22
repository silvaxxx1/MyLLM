"""
Configuration Management for Transformer-based Models

This module defines a `Config` class for managing the configuration settings of various transformer-based models, such as GPT-2 and LLaMA. The `Config` class provides functionality for:

1. **Core Parameters**:
    - `block_size`: Sequence length for input data.
    - `vocab_size`: The number of tokens in the vocabulary.
    - `n_layer`: Number of transformer layers.
    - `n_head`: Number of attention heads.
    - `n_embd`: Embedding dimensionality.
    - Other model-specific parameters.

2. **Architecture Variations**:
    - Options for different normalization layers (`LayerNorm`, `RMSNorm`).
    - Activation functions like `gelu`, `relu`, etc.
    - Customizable MLP configurations (`GptNeoxMLP`, `LLaMAMLP`, etc.).
    - Parallel residual connections, rotary embeddings, and other LLaMA-specific features.

3. **Hyperparameters**:
    - Dropout rate for regularization.
    - Learning rate, weight decay, and optimizer settings (e.g., Adam optimizer).
    - Additional model-specific hyperparameters like attention bias, dropout rates, and more.

4. **Configuration Management**:
    - Save and load configurations from JSON files for easy configuration management.
    - Dynamic updating of configuration parameters.

5. **Validation**:
    - Validation checks to ensure compatibility of certain parameters (e.g., ensuring that `n_embd` is divisible by `n_head` for correct attention behavior).

6. **Trainable Parameters**:
    - A method to retrieve only the trainable parameters from the configuration (those that are `int`, `float`, or `bool`).

7. **Multi-Model Support**:
    - The ability to manage multiple model configurations (e.g., GPT-2, LLaMA) using a configuration registry.

## Example Usage

- **Creating a Config instance**:  
    `Config.from_name("gpt2-small")` creates a `Config` instance for the GPT-2 small model.

- **Validating the configuration**:  
    `config.validate()` ensures that the configuration parameters are valid.

- **Saving and loading configurations**:  
    `config.save("config.json")` saves the configuration to a JSON file, while `Config.load("config.json")` loads it from a file.

- **Updating configuration parameters**:  
    `config.update(block_size=2048, learning_rate=1e-4)` dynamically updates specific parameters.

- **Getting trainable parameters**:  
    `config.get_trainable_params()` returns a dictionary of trainable parameters.

## Attributes
- `name`: Name of the model configuration.
- `block_size`: Size of each sequence block.
- `vocab_size`: The size of the vocabulary.
- `n_layer`: Number of transformer layers.
- `n_head`: Number of attention heads.
- `n_embd`: Dimensionality of the embedding layer.
- `norm_class_name`: Type of normalization layer used (either `LayerNorm` or `RMSNorm`).
- `activation`: Activation function used in the model.
- `learning_rate`: Learning rate for optimization.
- `weight_decay`: Weight decay used for regularization.
- `beta1`, `beta2`: Adam optimizer parameters.

## Configuration Registry

The module supports different configurations for various transformer models. You can access available configurations via the `available_configs()` method.

Example configurations include:
- `"gpt2-small"`, `"gpt2-medium"`, `"gpt2-large"`, `"gpt2-xl"`, `"llama2-7b"`, `"llama2-13b"`, etc.

---

The `Config` class is designed for managing large-scale transformer models with flexible settings, allowing easy integration into training pipelines.

"""


# Import statements
from dataclasses import dataclass, field
from typing import Optional, Any, Dict , Literal , Type
import json
from pathlib import Path 
import torch 

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
    head_size: Optional[int] = None  # Size of each attention head

    # Architecture variations
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"  # Type of normalization layer used (LayerNorm or RMSNorm)
    activation: str = "gelu"  # Activation function (gelu, relu, etc.)
    mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = "GptNeoxMLP"
    scale_embeddings: bool = False  # Whether to scale embeddings by sqrt(d_model)
    mlp_ratio: float = 4.0  # Ratio of hidden dimension to embedding dimension in the MLP
    lm_head_bias: bool = False
    attention_bias: bool = False  # Whether to use attention bias
    bias: bool = False
    mlp_hidden_size: Optional[int] = None
    post_mlp_norm: bool = False
    gelu_approx: str = "none"

    rotary_percentage: float = 0.0  # Percentage for rotary embeddings (specific to LLaMA models)
    parallel_residual: bool = False  # Whether to use parallel residual connections (specific to LLaMA)
    shared_attention_norm: bool = False  # Whether to use shared attention norm (specific to LLaMA)
    norm_eps: float = 1e-5  # Small epsilon for normalization (specific to LLaMA)
    n_query_groups: int = 32  # Number of query groups (specific to LLaMA)
    norm_qk: bool = False  # Whether to use normalized queries and keys
    use_rope: bool = False  # Whether to use rope embeddings
    rope_base: int = 10000

    attention_scores_scalar: Optional[int] = None
    softcapping_threshold: Optional[float] = None
    attention_logit_softcapping: Optional[float] = None
    post_attention_norm: bool = False

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

        # Ensure that head_size is set if not provided
        if self.head_size is None:
            self.head_size = self.n_embd // self.n_head  # Default to n_embd // n_head

        # Ensure that mlp_hidden_size is set if not provided
        if self.mlp_hidden_size is None:
            self.mlp_hidden_size = int(self.n_embd * self.mlp_ratio)  # Default to n_embd * mlp_ratio

        # Validate the configuration parameters after initialization
        self.validate()

    def __repr__(self):
        # Return a string representation of the config with key-value pairs
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"Config({params})"

    def save(self, file_path: str):
        """Save the configuration to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load(cls, file_path: str):
        """Load the configuration from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def update(self, **kwargs):
        """Update the configuration with new key-value pairs."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Invalid config key '{key}', skipping update.")

    def get_trainable_params(self):
        """Get a dictionary of trainable parameters (those that are int, float, or bool)."""
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, bool))}

    def validate(self):
        """Validate the configuration parameters."""
        # Ensure that n_embd is divisible by n_head for correct attention behavior
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        # Ensure that block_size is positive
        assert self.block_size > 0, "block_size must be positive"
        # Ensure that mlp_ratio is positive for proper scaling of the MLP layers
        assert self.mlp_ratio > 0, "mlp_ratio must be positive"
        print("✅ All checks passed.")

    @classmethod
    def available_configs(cls):
        """Return the list of available configurations by accessing the global configuration registry."""
        return list(name_to_config.keys())

    @classmethod
    def from_name(cls, name: str):
        """Create a Config instance from a configuration name."""
        if name not in name_to_config:
            raise ValueError(f"Config with name {name} not found.")
        return cls(**name_to_config[name])

    @property
    def mlp_class(self) -> Type:
        """
        Dynamically resolves the MLP class based on `mlp_class_name`.
        """
        import model  # Import the module where MLP classes are defined
        return getattr(model, self.mlp_class_name)

    @property
    def norm_class(self) -> Type:
        """
        Dynamically resolves the normalization class based on `norm_class_name`.
        Supports only `LayerNorm` and `RMSNorm`.
        """
        if self.norm_class_name == "RMSNorm":
            from model import RMSNorm  # Import RMSNorm from the appropriate module
            return RMSNorm  # Return the RMSNorm class directly

        if self.norm_class_name == "LayerNorm":
            return torch.nn.LayerNorm  # Return the LayerNorm class directly

        raise ValueError(f"Unsupported normalization class: {self.norm_class_name}")
    
#  google docs string this 
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

