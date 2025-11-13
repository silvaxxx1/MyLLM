"""
Configuration Management for Transformer-based Models

This module defines a `ModelConfig` class for managing the configuration settings 
of various transformer-based models, such as GPT-2 and LLaMA. The `ModelConfig` 
class provides functionality for:

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
    `ModelConfig.from_name("gpt2-small")` creates a `ModelConfig` instance for the GPT-2 small model.

- **Validating the configuration**:  
    `config.validate()` ensures that the configuration parameters are valid.

- **Saving and loading configurations**:  
    `config.save("config.json")` saves the configuration to a JSON file, while `ModelConfig.load("config.json")` loads it from a file.

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

The `ModelConfig` class is designed for managing large-scale transformer models with flexible settings, allowing easy integration into training pipelines.

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
    """
    Configuration class for transformer-based language models.
    
    This class encapsulates all the parameters needed to configure a transformer model,
    including architectural details, training hyperparameters, and model-specific settings.
    
    Attributes:
        name (str): Name identifier for the configuration
        block_size (int): Maximum sequence length the model can handle
        vocab_size (int): Size of the token vocabulary
        padded_vocab_size (Optional[int]): Vocabulary size after padding (for alignment)
        n_layer (int): Number of transformer layers/blocks
        n_head (int): Number of attention heads
        n_embd (int): Embedding dimension size
        eps (float): Epsilon value for numerical stability
        head_size (Optional[int]): Size of each attention head (calculated if None)
        
        # Normalization and Activation
        norm_class_name (Literal): Type of normalization ("LayerNorm" or "RMSNorm")
        activation (str): Activation function ("gelu", "relu", "silu", etc.)
        mlp_class_name (Literal): Type of MLP implementation
        scale_embeddings (bool): Whether to scale embeddings by sqrt(n_embd)
        mlp_ratio (float): Ratio for MLP hidden size expansion (e.g., 4.0 for 4x expansion)
        lm_head_bias (bool): Whether to include bias in language model head
        attention_bias (bool): Whether to include bias in attention layers
        bias (bool): Global bias setting for linear layers
        mlp_hidden_size (Optional[int]): Hidden size for MLP (calculated if None)
        post_mlp_norm (bool): Whether to apply normalization after MLP
        gelu_approx (str): GELU approximation type ("none", "tanh")
        position_embedding (str): Type of position embedding ("learned", "rope", "none")
        
        # Attention Configuration
        causal_attention (bool): Whether to use causal attention masking
        rotary_percentage (float): Percentage of dimensions to apply rotary embeddings to
        parallel_residual (bool): Whether to use parallel residual connections (LLaMA style)
        shared_attention_norm (bool): Whether to share normalization between attention paths
        norm_eps (float): Epsilon value for normalization layers
        n_query_groups (Optional[int]): Number of query groups for grouped-query attention
        norm_qk (bool): Whether to normalize queries and keys
        use_rope (bool): Whether to use rotary position embeddings
        rope_base (int): Base value for rotary position embeddings
        
        # Advanced Attention Settings
        attention_scores_scalar (Optional[int]): Scalar for attention scores
        softcapping_threshold (Optional[float]): Threshold for attention softcapping
        attention_logit_softcapping (Optional[float]): Softcapping value for attention logits
        post_attention_norm (bool): Whether to apply normalization after attention
        
        # Weight Management
        weight_tying (bool): Whether to tie weights between input and output embeddings
        learnable_pos_emb (bool): Whether position embeddings are learnable
        
        # Training Hyperparameters
        dropout (float): Dropout rate for regularization
        learning_rate (float): Learning rate for optimization
        weight_decay (float): Weight decay for regularization
        beta1 (float): Adam optimizer beta1 parameter
        beta2 (float): Adam optimizer beta2 parameter
        
        # Model Loading and Device Settings
        load_in_8bit (bool): Whether to load model in 8-bit quantization
        load_in_4bit (bool): Whether to load model in 4-bit quantization
        torch_dtype (Optional[str]): PyTorch data type ("float32", "float16", "bfloat16")
        low_cpu_mem_usage (bool): Whether to optimize for low CPU memory usage
        device_map (Optional[str]): Device mapping strategy ("auto", "cpu", "cuda")
        
        # Additional Parameters
        extra_params (Dict[str, Any]): Extra parameters for model-specific configurations
    """
    
    # Core Model Architecture
    name: str = ""
    block_size: int = 1024
    vocab_size: int = 50257
    padded_vocab_size: Optional[int] = None
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    eps: float = 1e-5
    head_size: Optional[int] = None

    # Normalization and Activation
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
    position_embedding: str = "learned"  # "learned" or "rope" or "none"
    intermediate_size: Optional[int] = None  # For GPT-NeoX style MLP

    # Attention Configuration
    causal_attention: bool = True
    rotary_percentage: float = 0.0
    parallel_residual: bool = False
    shared_attention_norm: bool = False
    norm_eps: float = 1e-5
    n_query_groups: Optional[int] = None
    norm_qk: bool = False
    use_rope: bool = False
    rope_base: int = 10000

    # Advanced Attention Settings
    attention_scores_scalar: Optional[int] = None
    softcapping_threshold: Optional[float] = None
    attention_logit_softcapping: Optional[float] = None
    post_attention_norm: bool = False
    
    # Weight Management
    weight_tying: bool = True  # Whether to tie the weights of lm_head and token embeddings
    learnable_pos_emb: bool = True  # Whether to use learnable positional embeddings

    # Training Hyperparameters
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999

    # Model Loading and Device Settings
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: Optional[str] = None  # "float32", "float16", "bfloat16"
    low_cpu_mem_usage: bool = True
    device_map: Optional[str] = None  # "auto", "cpu", "cuda"

    # Additional Parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Initialize derived configuration parameters after main initialization.
        
        This method calculates dependent parameters that are not explicitly provided
        but can be derived from other configuration values. It also validates the
        configuration to ensure parameter compatibility.
        """
        # Set padded_vocab_size to vocab_size if not specified
        if self.padded_vocab_size is None:
            self.padded_vocab_size = self.vocab_size
            
        # Calculate head_size from embedding dimension and number of heads
        if self.head_size is None:
            self.head_size = self.n_embd // self.n_head
            
        # Calculate MLP hidden size from embedding dimension and expansion ratio
        if self.mlp_hidden_size is None:
            self.mlp_hidden_size = int(self.n_embd * self.mlp_ratio)
            
        # Set query groups to number of heads if not specified (standard multi-head attention)
        if self.n_query_groups is None:
            self.n_query_groups = self.n_head
            
        # Set intermediate_size to mlp_hidden_size if not specified
        if self.intermediate_size is None:
            self.intermediate_size = self.mlp_hidden_size
            
        # Validate the configuration parameters
        self.validate()

    def __repr__(self) -> str:
        """
        Return a string representation of the configuration.
        
        Returns:
            str: String representation showing all configuration parameters
        """
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"ModelConfig({params})"

    def save(self, file_path: str) -> None:
        """
        Save the configuration to a JSON file.
        
        Args:
            file_path (str): Path to the JSON file where configuration will be saved
            
        Example:
            >>> config.save("my_model_config.json")
        """
        with open(file_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load(cls, file_path: str) -> 'ModelConfig':
        """
        Load a configuration from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file containing the configuration
            
        Returns:
            ModelConfig: Loaded configuration instance
            
        Example:
            >>> config = ModelConfig.load("my_model_config.json")
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def update(self, **kwargs) -> None:
        """
        Update configuration parameters with new values.
        
        Args:
            **kwargs: Key-value pairs of parameters to update
            
        Example:
            >>> config.update(learning_rate=1e-4, block_size=2048)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Invalid config key '{key}', skipping update.")

    def get_trainable_params(self) -> Dict[str, Any]:
        """
        Get a dictionary of trainable parameters (int, float, bool).
        
        Returns:
            Dict[str, Any]: Dictionary containing only trainable parameters
            
        Example:
            >>> trainable_params = config.get_trainable_params()
        """
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, bool))}

    def validate(self) -> None:
        """
        Validate the configuration parameters for compatibility.
        
        This method checks that critical parameter relationships are maintained,
        such as ensuring the embedding dimension is divisible by the number of
        attention heads.
        
        Raises:
            AssertionError: If any validation check fails
        """
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.block_size > 0, "block_size must be positive"
        assert self.mlp_ratio > 0, "mlp_ratio must be positive"

    def estimate_memory(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> Dict[str, float]:
        """
        Estimate memory usage in gigabytes for the configured model.
        
        This provides a rough estimate of memory requirements for both parameters
        and activations during inference or training.
        
        Args:
            batch_size (int): Batch size for memory estimation
            dtype (torch.dtype): Data type for parameter storage
            
        Returns:
            Dict[str, float]: Dictionary containing memory estimates with keys:
                - "parameters_gb": Memory for model parameters
                - "activations_gb": Memory for activations
                - "total_gb": Total memory requirement
                - "n_parameters": Total number of parameters
                
        Example:
            >>> memory_estimate = config.estimate_memory(batch_size=2, dtype=torch.float16)
        """
        # Determine bytes per parameter based on data type
        bytes_per_param = 4 if dtype == torch.float32 else 2  # float32 vs float16/bfloat16
        
        # Calculate total number of parameters
        n_params = (
            self.n_layer * (
                # Self-attention parameters (QKV + output projection)
                4 * self.n_embd * self.n_embd +
                # MLP parameters (two linear transformations)
                2 * self.n_embd * self.mlp_hidden_size +
                self.mlp_hidden_size * self.n_embd
            ) +
            # Embedding parameters
            self.vocab_size * self.n_embd +
            # Layer normalization parameters
            4 * self.n_layer * self.n_embd
        )
        
        # Calculate parameter memory in GB
        param_memory = n_params * bytes_per_param / 1024**3
        
        # Rough estimate of activation memory
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
    def from_name(cls, name: str) -> 'ModelConfig':
        """
        Create a configuration instance from a predefined configuration name.
        
        Args:
            name (str): Name of the predefined configuration
            
        Returns:
            ModelConfig: Configuration instance for the specified model
            
        Raises:
            ValueError: If the configuration name is not found
            
        Example:
            >>> config = ModelConfig.from_name("gpt2-small")
        """
        if name not in name_to_config:
            raise ValueError(f"Configuration with name '{name}' not found. Available configurations: {list(name_to_config.keys())}")
        
        config = cls(**name_to_config[name])
        config.__post_init__()  # Ensure all derived fields are properly set
        return config

    @classmethod
    def available_configs(cls) -> list:
        """
        Get a list of all available predefined configuration names.
        
        Returns:
            list: List of available configuration names
            
        Example:
            >>> available = ModelConfig.available_configs()
            >>> print(available)
            ['gpt2-small', 'gpt2-medium', ...]
        """
        return list(name_to_config.keys())

    @property
    def mlp_class(self) -> Type:
        """
        Get the MLP class based on the configuration.
        
        Returns:
            Type: The MLP class to use
            
        Raises:
            AttributeError: If the MLP class is not found in the model module
        """
        import myllm.model as model
        return getattr(model, self.mlp_class_name)

    @property
    def norm_class(self) -> Type:
        """
        Get the normalization class based on the configuration.
        
        Returns:
            Type: The normalization class to use (LayerNorm or RMSNorm)
            
        Raises:
            ValueError: If the normalization class name is not supported
        """
        if self.norm_class_name == "RMSNorm":
            return torch.nn.RMSNorm
        if self.norm_class_name == "LayerNorm":
            return torch.nn.LayerNorm
        raise ValueError(f"Unsupported normalization class: {self.norm_class_name}")


# Configuration Registry with Updated GPT-2 Configurations
name_to_config = {
    cfg["name"]: cfg for cfg in [
        # GPT-2 Configurations with Complete Settings
        dict(
            name="gpt2-small", 
            block_size=1024, 
            vocab_size=50257, 
            n_layer=12, 
            n_head=12, 
            n_embd=768, 
            norm_class_name="LayerNorm", 
            mlp_class_name="GptMLP", 
            activation="gelu", 
            scale_embeddings=False,  # GPT-2 doesn't scale embeddings
            position_embedding="learned",  # GPT-2 uses learned position embeddings
            weight_tying=True,  # GPT-2 ties weights between wte and lm_head
            use_rope=False,  # GPT-2 does NOT use RoPE
            attention_bias=True,  # GPT-2 uses bias in attention
            bias=True,  # GPT-2 uses bias in linear layers
            lm_head_bias=False,  # GPT-2 lm_head does NOT have bias
            causal_attention=True,  # GPT-2 is causal
            parallel_residual=False,  # GPT-2 uses sequential residuals
            norm_eps=1e-5,  # GPT-2 LayerNorm epsilon
            mlp_ratio=4.0,  # Standard GPT MLP expansion
            gelu_approx="none"  # GPT-2 uses exact GELU
        ),
        dict(
            name="gpt2-medium", 
            block_size=1024, 
            vocab_size=50257, 
            n_layer=24, 
            n_head=16, 
            n_embd=1024,
            norm_class_name="LayerNorm", 
            mlp_class_name="GptMLP", 
            activation="gelu",
            scale_embeddings=False,
            position_embedding="learned",
            weight_tying=True,
            use_rope=False,
            attention_bias=True,
            bias=True,
            lm_head_bias=False,
            causal_attention=True,
            parallel_residual=False,
            norm_eps=1e-5,
            mlp_ratio=4.0,
            gelu_approx="none"
        ),
        dict(
            name="gpt2-large", 
            block_size=1024, 
            vocab_size=50257, 
            n_layer=36, 
            n_head=20, 
            n_embd=1280,
            norm_class_name="LayerNorm", 
            mlp_class_name="GptMLP", 
            activation="gelu",
            scale_embeddings=False,
            position_embedding="learned",
            weight_tying=True,
            use_rope=False,
            attention_bias=True,
            bias=True,
            lm_head_bias=False,
            causal_attention=True,
            parallel_residual=False,
            norm_eps=1e-5,
            mlp_ratio=4.0,
            gelu_approx="none"
        ),
        dict(
            name="gpt2-xl", 
            block_size=1024, 
            vocab_size=50257, 
            n_layer=48, 
            n_head=25, 
            n_embd=1600,
            norm_class_name="LayerNorm", 
            mlp_class_name="GptMLP", 
            activation="gelu",
            scale_embeddings=False,
            position_embedding="learned",
            weight_tying=True,
            use_rope=False,
            attention_bias=True,
            bias=True,
            lm_head_bias=False,
            causal_attention=True,
            parallel_residual=False,
            norm_eps=1e-5,
            mlp_ratio=4.0,
            gelu_approx="none"
        ),
        # LLaMA Configurations
        dict(
            name="llama2-7b", 
            block_size=4096, 
            vocab_size=32000, 
            n_layer=32, 
            n_head=32, 
            n_embd=4096, 
            norm_class_name="RMSNorm", 
            mlp_class_name="LLaMAMLP", 
            rotary_percentage=1.0, 
            parallel_residual=True, 
            norm_eps=1e-5,
            position_embedding="rope",  # LLaMA uses RoPE
            use_rope=True,
            weight_tying=False  # LLaMA doesn't use weight tying
        ),
        dict(
            name="llama2-13b", 
            block_size=4096, 
            vocab_size=32000, 
            n_layer=40, 
            n_head=40, 
            n_embd=5120, 
            norm_class_name="RMSNorm", 
            mlp_class_name="LLaMAMLP", 
            rotary_percentage=1.0, 
            parallel_residual=True, 
            norm_eps=1e-5,
            position_embedding="rope",
            use_rope=True,
            weight_tying=False
        ),
        dict(
            name="llama3-1b", 
            block_size=8192, 
            vocab_size=128256, 
            n_layer=24, 
            n_head=16, 
            n_embd=2048, 
            norm_class_name="RMSNorm", 
            mlp_class_name="LLaMAMLP", 
            rotary_percentage=1.0, 
            parallel_residual=True, 
            norm_eps=1e-5,
            position_embedding="rope",
            use_rope=True,
            weight_tying=False
        ),
        dict(
            name="llama3-3b", 
            block_size=8192, 
            vocab_size=128256, 
            n_layer=32, 
            n_head=32, 
            n_embd=3072, 
            norm_class_name="RMSNorm", 
            mlp_class_name="LLaMAMLP", 
            rotary_percentage=1.0, 
            parallel_residual=True, 
            norm_eps=1e-5,
            position_embedding="rope",
            use_rope=True,
            weight_tying=False
        ),
        dict(
            name="llama3-8b", 
            block_size=8192, 
            vocab_size=128256, 
            n_layer=32, 
            n_head=32, 
            n_embd=4096, 
            norm_class_name="RMSNorm", 
            mlp_class_name="LLaMAMLP", 
            rotary_percentage=1.0, 
            parallel_residual=True, 
            norm_eps=1e-5,
            position_embedding="rope",
            use_rope=True,
            weight_tying=False
        )
    ]
}