# trainer/configs/base_config.py
from abc import ABC
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
class BaseTrainerConfig(ABC):
    """Base configuration class for all trainers"""
    # Model and data
    model_name_or_path: str = "gpt2"
    dataset_name: Optional[str] = None
    data_path: Optional[str] = None
    output_dir: str = "./output"
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    
    # Optimizer and scheduler
    optimizer: OptimizerType = OptimizerType.ADAMW
    scheduler: SchedulerType = SchedulerType.LINEAR
    weight_decay: float = 0.01
    
    # Logging and evaluation
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Monitoring and Reporting
    logging_backend: LoggingBackend = LoggingBackend.WANDB
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    log_model: bool = True
    log_predictions: bool = False
    
    # WandB specific configs
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_config_exclude: List[str] = field(default_factory=list)
    wandb_resume: Union[bool, str] = False
    
    # TensorBoard configs
    tensorboard_log_dir: Optional[str] = None
    
    # Metric tracking
    metric_for_best_model: Optional[str] = None
    greater_is_better: bool = True
    load_best_model_at_end: bool = False
    
    # Hardware and performance
    device: str = "auto"
    mixed_precision: bool = True
    dataloader_num_workers: int = 4
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    local_rank: int = -1
    
    # DeepSpeed integration
    deepspeed_config: Optional[Dict[str, Any]] = None
    
    # Random seed
    seed: int = 42
    
    # Additional config for extensibility
    extra_config: Dict[str, Any] = field(default_factory=dict)




