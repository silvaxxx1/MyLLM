# =====================================
# configs/base_config.py
"""
Base Training Configuration

Contains BaseTrainConfig class with common parameters shared across
all training methods (SFT, DPO, PPO).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
import json
import os
from pathlib import Path

@dataclass
class BaseTrainConfig:
    """Base configuration class with common training parameters"""
    
    # Model and generation configs
    model_config: Optional[Any] = None  # ModelConfig instance
    generation_config: Optional[Any] = None  # GenerationConfig instance
    
    # Training basics
    max_epochs: int = 3
    max_steps: Optional[int] = None  # If set, overrides max_epochs
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimizer settings
    optimizer: Literal["adamw", "sgd", "adafactor"] = "adamw"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler settings
    scheduler: Literal["linear", "cosine", "cosine_with_restarts", "constant"] = "linear"
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    
    # Logging and evaluation
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    eval_strategy: Literal["steps", "epoch", "no"] = "steps"
    save_strategy: Literal["steps", "epoch", "no"] = "steps"
    
    # Data settings
    max_length: int = 2048
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Hardware and performance
    device: str = "auto"  # "auto", "cuda", "cpu"
    mixed_precision: Literal["fp16", "bf16", "no"] = "bf16"
    gradient_checkpointing: bool = True
    dataloader_drop_last: bool = True
    
    # Paths
    output_dir: str = "./outputs"
    logging_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    # Monitoring
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # Checkpointing
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set effective batch size
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps

    def save(self, file_path: str):
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, handling nested configs"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):  # Nested config object
                result[key] = value.__dict__ if value else None
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary"""
        return cls(**config_dict)

    @classmethod
    def load(cls, file_path: str):
        """Load configuration from JSON file"""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)