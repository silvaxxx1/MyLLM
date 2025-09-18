
# trainer/configs/trainer_config.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum

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

@dataclass
class TrainerConfig:
    """Training configuration compatible with existing ModelConfig"""
    # Model configuration (will use existing ModelConfig)
    model_config_name: str = "gpt2-small"  # Use ModelConfig.from_name()
    model_config_path: Optional[str] = None  # Or load from file
    output_dir: str = "./output"
    
    # Data
    dataset_name: Optional[str] = None
    data_path: Optional[str] = None
    tokenizer_name: str = "gpt2"  # Compatible with your tokenizer factory
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    
    # Optimizer settings (will override ModelConfig if specified)
    learning_rate: Optional[float] = None  # Use ModelConfig.learning_rate if None
    weight_decay: Optional[float] = None   # Use ModelConfig.weight_decay if None
    beta1: Optional[float] = None          # Use ModelConfig.beta1 if None
    beta2: Optional[float] = None          # Use ModelConfig.beta2 if None
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    scheduler_type: SchedulerType = SchedulerType.LINEAR
    
    # Logging and evaluation
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Monitoring
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    log_model: bool = True
    log_predictions: bool = False
    
    # WandB configuration
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_resume: Union[bool, str] = False
    
    # TensorBoard
    tensorboard_log_dir: Optional[str] = None
    
    # Metric tracking
    metric_for_best_model: Optional[str] = None
    greater_is_better: bool = True
    load_best_model_at_end: bool = False
    
    # Hardware
    device: str = "auto"
    mixed_precision: bool = True
    dataloader_num_workers: int = 4
    
    # System
    seed: int = 42
    
    # Training specific
    max_seq_length: Optional[int] = None  # Use ModelConfig.block_size if None
    preprocessing_num_workers: int = 4
    resume_from_checkpoint: Optional[str] = None