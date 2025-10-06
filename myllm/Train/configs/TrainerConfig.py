# trainer/configs/trainer_config.py
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)

# ----------------------------
# Enums for common options
# ----------------------------
class LoggingBackend(Enum):
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
    NONE = "none"

class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"

class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    CONSTANT = "constant"

class DeviceType(Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"

# ----------------------------
# Base Trainer Config
# ----------------------------
@dataclass
class TrainerConfig:
    # ----------------------------
    # Model
    # ----------------------------
    model_config_name: str = "gpt2-small"
    tokenizer_name: str = "gpt2"
    max_seq_length: Optional[int] = None
    model_config_path: Optional[str] = None

    # ----------------------------
    # Data
    # ----------------------------
    dataset_name: Optional[str] = None
    data_path: Optional[str] = None
    preprocessing_num_workers: int = 4

    # ----------------------------
    # Training
    # ----------------------------
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    seed: int = 42

    # ----------------------------
    # Optimizer / Scheduler
    # ----------------------------
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    scheduler_type: SchedulerType = SchedulerType.LINEAR
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    beta1: Optional[float] = None
    beta2: Optional[float] = None

    # ----------------------------
    # Logging / Monitoring
    # ----------------------------
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    metric_for_best_model: str = "eval_loss"  # ✅ Changed from Optional to default
    greater_is_better: bool = False  # ✅ Changed to False for loss metrics
    load_best_model_at_end: bool = False
    output_dir: str = "./output"

    # ----------------------------
    # WandB Specific Settings (ADD THESE)
    # ----------------------------
    wandb_project: Optional[str] = "myllm-training"
    wandb_run_name: Optional[str] = None
    wandb_notes: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_entity: Optional[str] = None  # Your wandb username/team

    # ----------------------------
    # Hardware / System
    # ----------------------------
    device: Union[DeviceType, str] = DeviceType.AUTO  # ✅ Allow string or enum
    mixed_precision: bool = True
    dataloader_num_workers: int = 4
    use_compile: bool = False
    use_deepspeed: bool = False
    deepspeed_config_path: Optional[str] = None

    # ----------------------------
    # Checkpoint / Resume
    # ----------------------------
    resume_from_checkpoint: Optional[str] = None

    def __post_init__(self):
        """Post-initialization validation and setup"""
        logger.info("Initializing TrainerConfig...")
        
        # Handle device conversion if it's a string
        if isinstance(self.device, str):
            try:
                self.device = DeviceType(self.device.lower())
            except ValueError:
                raise ValueError(f"Invalid device: {self.device}. Must be one of: {[e.value for e in DeviceType]}")
        
        # Set default values for optional parameters
        if self.learning_rate is None:
            self.learning_rate = 5e-5
            logger.info(f"Using default learning_rate: {self.learning_rate}")
            
        if self.weight_decay is None:
            self.weight_decay = 0.01
            logger.info(f"Using default weight_decay: {self.weight_decay}")
            
        if self.beta1 is None:
            self.beta1 = 0.9
        if self.beta2 is None:
            self.beta2 = 0.999
        
        # Basic validation
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.use_deepspeed and (not self.deepspeed_config_path or not os.path.exists(self.deepspeed_config_path)):
            raise ValueError("use_deepspeed is True but deepspeed_config_path is invalid")
        
        # Validate WandB settings
        if "wandb" in self.report_to and not self.wandb_project:
            raise ValueError("wandb_project must be set when using wandb logging")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def validate(self):
        """Legacy validation method - now calls __post_init__ for backward compatibility"""
        self.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, handling enums properly"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value
            elif hasattr(value, '__dict__'):
                config_dict[key] = asdict(value)
            else:
                config_dict[key] = value
        return config_dict

    def get_wandb_config(self) -> Dict[str, Any]:
        """Get config specifically for WandB initialization"""
        return {
            "model_config_name": self.model_config_name,
            "tokenizer_name": self.tokenizer_name,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "warmup_steps": self.warmup_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer_type": self.optimizer_type.value if isinstance(self.optimizer_type, Enum) else self.optimizer_type,
            "scheduler_type": self.scheduler_type.value if isinstance(self.scheduler_type, Enum) else self.scheduler_type,
        }