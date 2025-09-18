# trainer/configs/trainer_config.py
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import os

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
    """
    Base trainer configuration
    Includes grouped sections: model, data, training, optimizer, logging, hardware
    """

    # ----------------------------
    # Model Configuration
    # ----------------------------
    model_config_name: str = "gpt2-small"
    model_config_path: Optional[str] = None
    tokenizer_name: str = "gpt2"
    max_seq_length: Optional[int] = None  # Uses ModelConfig.block_size if None

    # ----------------------------
    # Data / Dataset
    # ----------------------------
    dataset_name: Optional[str] = None
    data_path: Optional[str] = None
    preprocessing_num_workers: int = 4

    # ----------------------------
    # Training Hyperparameters
    # ----------------------------
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    max_seq_length: Optional[int] = None  # Overrides model block_size if provided

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
    log_model: bool = True
    log_predictions: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: bool = True
    load_best_model_at_end: bool = False

    # WandB specific
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_resume: Union[bool, str] = False

    # TensorBoard
    tensorboard_log_dir: Optional[str] = None

    # ----------------------------
    # Hardware / System
    # ----------------------------
    device: DeviceType = DeviceType.AUTO
    mixed_precision: bool = True
    dataloader_num_workers: int = 4
    seed: int = 42

    # ----------------------------
    # Multi-GPU / DeepSpeed
    # ----------------------------
    use_deepspeed: bool = False
    deepspeed_config_path: Optional[str] = None
    distributed_backend: str = "nccl"
    local_rank: int = -1  # For DDP
    
    # ----------------------------
    # Checkpoint / Resume
    # ----------------------------
    resume_from_checkpoint: Optional[str] = None

    # ----------------------------
    # Validation Method
    # ----------------------------
    def validate(self):
        if self.device not in DeviceType:
            raise ValueError(f"Invalid device: {self.device}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.use_deepspeed and (self.deepspeed_config_path is None or not os.path.exists(self.deepspeed_config_path)):
            raise ValueError("use_deepspeed is True but deepspeed_config_path is invalid")
        # Add more checks as needed

    # Optional: convert to dict safely for logging or saving
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
