from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import json

@dataclass
class Config:
    # Core parameters
    name: str = ""
    block_size: int = 1024
    vocab_size: int = 50257
    padded_vocab_size: Optional[int] = None
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    eps: float = 1e-5

    # Architecture variations
    norm_class_name: str = "LayerNorm"
    activation: str = "gelu"
    mlp_class_name: str = "GptMLP"
    scale_embeddings: bool = False
    mlp_ratio: float = 4.0

    # LLama 
    rotary_percentage: float = 0.0
    parallel_residual: bool = False
    norm_eps: float = 1e-5

    # Hyperparameters
    dropout: float = 0.1
    bias: bool = False
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999

    # Extra parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = self.vocab_size
        
        self.validate()

    def __repr__(self):
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
        """ Get a dictionary of trainable parameters. """
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, bool))}

    def validate(self):
        """ Validate the configuration parameters. """
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.block_size > 0, "block_size must be positive"
        assert self.mlp_ratio > 0, "mlp_ratio must be positive"
        print("✅ All checks passed.")

    @classmethod
    def available_configs(cls):
        """ Return the list of available configurations. """
        return list(name_to_config.keys())

    @classmethod
    def from_name(cls, name: str):
        """ Create a Config instance from a configuration name. """
        if name not in name_to_config:
            raise ValueError(f"Config with name {name} not found.")
        return cls(**name_to_config[name])


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

# Test the functionality of the Config class and methods

# 1. Test creating Config instances for all available models
for model_name in name_to_config:
    print(f"Testing config for {model_name}...")
    config = Config.from_name(model_name)
    print(config)

# 2. Test saving and loading configurations
config_to_test = Config.from_name("gpt2-small")
config_to_test.save("gpt2-small-config.json")
loaded_config = Config.load("gpt2-small-config.json")
print(f"Loaded config for {config_to_test.name}: {loaded_config}")

# 3. Test updating a configuration
updated_config = Config.from_name("gpt2-small")
updated_config.update(n_embd=1024, n_layer=18)
print(f"Updated config for {updated_config.name}: {updated_config}")

# 4. Test validation to make sure no errors occur for valid configs
for model_name in name_to_config:
    print(f"Validating config for {model_name}...")
    config = Config.from_name(model_name)
    try:
        config.validate()
    except AssertionError as e:
        print(f"Validation failed for {model_name}: {e}")

# 5. Test getting trainable parameters
trainable_params = config.get_trainable_params()
print(f"Trainable parameters for {config.name}: {trainable_params}")

# 6. Test accessing all available configurations
available_configs = Config.available_configs()
print(f"Available configurations: {available_configs}")
