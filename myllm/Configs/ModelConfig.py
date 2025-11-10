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

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Literal, Type
import json
import torch
import logging

# Set up logging configuration
logging.basicConfig(level=logging.WARNING)  # This suppresses info level messages

@dataclass
class ModelConfig:
    name: str = ""
    block_size: int = 1024
    vocab_size: int = 50257
    padded_vocab_size: Optional[int] = None
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    eps: float = 1e-5
    head_size: Optional[int] = None

    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    activation: str = "gelu"
    mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE", "GptMLP"] = "GptNeoxMLP"
    scale_embeddings: bool = False
    mlp_ratio: float = 4.0
    lm_head_bias: bool = False
    attention_bias: bool = False
    bias: bool = False
    mlp_hidden_size: Optional[int] = None
    post_mlp_norm: bool = False
    gelu_approx: str = "none"

    causal_attention: bool = True
    rotary_percentage: float = 0.0
    parallel_residual: bool = False
    shared_attention_norm: bool = False
    norm_eps: float = 1e-5
    n_query_groups: Optional[int] = None  # Default value; will be updated in __post_init__ if necessary
    norm_qk: bool = False
    use_rope: bool = False
    rope_base: int = 10000

    attention_scores_scalar: Optional[int] = None
    softcapping_threshold: Optional[float] = None
    attention_logit_softcapping: Optional[float] = None
    post_attention_norm: bool = False
    weight_tying = True  # Whether to tie the weights of lm_head and token embeddings
    learnable_pos_emb = True  # Whether to use learnable positional embeddings

    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999

    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: Optional[str] = None  # "float32", "float16", "bfloat16"
    low_cpu_mem_usage: bool = True
    device_map: Optional[str] = None  # "auto", "cpu", "cuda"


    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = self.vocab_size
        if self.head_size is None:
            self.head_size = self.n_embd // self.n_head
        if self.mlp_hidden_size is None:
            self.mlp_hidden_size = int(self.n_embd * self.mlp_ratio)
        if self.n_query_groups is None:
            self.n_query_groups = self.n_head  # Ensure n_query_groups is equal to n_head
        self.validate()

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"Config({params})"

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Invalid config key '{key}', skipping update.")

    def get_trainable_params(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, bool))}

    def validate(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.block_size > 0, "block_size must be positive"
        assert self.mlp_ratio > 0, "mlp_ratio must be positive"
        # Removed the print statement here
        # logging.info("✅ All checks passed.")  # Use logging instead of print

    def estimate_memory(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> Dict[str, float]:
        """Estimate memory usage in GB"""
        bytes_per_param = 4 if dtype == torch.float32 else 2  # float32 vs float16/bfloat16
        
        # Parameter memory
        n_params = (
            self.n_layer * (
                # Self-attention
                4 * self.n_embd * self.n_embd +  # QKV + output projection
                # MLP
                2 * self.n_embd * self.mlp_hidden_size +
                self.mlp_hidden_size * self.n_embd
            ) +
            # Embeddings
            self.vocab_size * self.n_embd +
            # Layer norms
            4 * self.n_layer * self.n_embd
        )
        param_memory = n_params * bytes_per_param / 1024**3  # Convert to GB
        
        # Activation memory (rough estimate)
        activation_memory = (
            batch_size * self.block_size * self.n_embd * 
            self.n_layer * 4 * bytes_per_param / 1024**3
        )
        
        return {
            "parameters_gb": param_memory,
            "activations_gb": activation_memory,
            "total_gb": param_memory + activation_memory,
            "n_parameters": n_params
        }

    @classmethod
    def from_name(cls, name: str):
        if name not in name_to_config:
            raise ValueError(f"Config with name {name} not found.")
        config = cls(**name_to_config[name])
        config.__post_init__()  # ✅ Ensure all derived fields are set
        return config


    @classmethod
    def available_configs(cls):
        return list(name_to_config.keys())

    @property
    def mlp_class(self) -> Type:
        import myllm.model as model
        return getattr(model, self.mlp_class_name)

    @property
    def norm_class(self) -> Type:
        if self.norm_class_name == "RMSNorm":
            return torch.nn.RMSNorm
        if self.norm_class_name == "LayerNorm":
            return torch.nn.LayerNorm
        raise ValueError(f"Unsupported normalization class: {self.norm_class_name}")


# Configuration Registry
name_to_config = {
    cfg["name"]: cfg for cfg in [
        dict(name="gpt2-small", block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, norm_class_name="LayerNorm", mlp_class_name="GptMLP", activation="gelu", scale_embeddings=True),
        dict(name="gpt2-medium", block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024, norm_class_name="LayerNorm", mlp_class_name="GptMLP", activation="gelu", scale_embeddings=True),
        dict(name="gpt2-large", block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280, norm_class_name="LayerNorm", mlp_class_name="GptMLP", activation="gelu", scale_embeddings=True),
        dict(name="gpt2-xl", block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600, norm_class_name="LayerNorm", mlp_class_name="GptMLP", activation="gelu", scale_embeddings=True),
        dict(name="llama2-7b", block_size=4096, vocab_size=32000, n_layer=32, n_head=32, n_embd=4096, norm_class_name="RMSNorm", mlp_class_name="LLaMAMLP", rotary_percentage=1.0, parallel_residual=True, norm_eps=1e-5),
        dict(name="llama2-13b", block_size=4096, vocab_size=32000, n_layer=40, n_head=40, n_embd=5120, norm_class_name="RMSNorm", mlp_class_name="LLaMAMLP", rotary_percentage=1.0, parallel_residual=True, norm_eps=1e-5),
        dict(name="llama3-1b", block_size=8192, vocab_size=128256, n_layer=24, n_head=16, n_embd=2048, norm_class_name="RMSNorm", mlp_class_name="LLaMAMLP", rotary_percentage=1.0, parallel_residual=True, norm_eps=1e-5),
        dict(name="llama3-3b", block_size=8192, vocab_size=128256, n_layer=32, n_head=32, n_embd=3072, norm_class_name="RMSNorm", mlp_class_name="LLaMAMLP", rotary_percentage=1.0, parallel_residual=True, norm_eps=1e-5),
        dict(name="llama3-8b", block_size=8192, vocab_size=128256, n_layer=32, n_head=32, n_embd=4096, norm_class_name="RMSNorm", mlp_class_name="LLaMAMLP", rotary_percentage=1.0, parallel_residual=True, norm_eps=1e-5)
    ]
}


