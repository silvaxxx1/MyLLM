# train_config.py

from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
from enum import Enum

class TrainerType(Enum):
    SFT = "sft"
    DPO = "dpo" 
    PPO = "ppo"
    GRPO = "grpo"

class LRSchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"

@dataclass
class TrainConfig:
    """Base training configuration that can be extended for different training methods."""
    
    # === Training Type ===
    trainer_type: TrainerType = TrainerType.SFT
    
    # === General training parameters ===
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    run_name: Optional[str] = None  # Useful for organizing experiments

    num_train_epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    dataloader_drop_last: bool = True  # Important for consistent batch sizes

    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    max_grad_norm: float = 1.0
    max_steps: Optional[int] = None  # Overrides epochs if set

    warmup_steps: int = 0
    warmup_ratio: Optional[float] = None  # If set, overrides warmup_steps

    # === Device & precision ===
    fp16: bool = False
    bf16: bool = False
    device: Optional[str] = None  # auto-detect if None
    
    # === Model and tokenizer ===
    model_name_or_path: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    model_max_length: Optional[int] = None  # Max sequence length
    
    # === Data parameters ===
    max_prompt_length: Optional[int] = None  # For RL methods
    max_response_length: Optional[int] = None  # For RL methods
    truncation_mode: str = "keep_end"  # or "keep_start"
    
    # === Logging and checkpointing ===
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: Optional[int] = 3
    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None

    # === Evaluation ===
    evaluation_strategy: str = "steps"  # options: 'no', 'steps', 'epoch'
    eval_delay: int = 0
    eval_accumulation_steps: Optional[int] = None
    do_eval: bool = True

    # === Early stopping ===
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None

    # === Data loading ===
    num_workers: int = 4
    pin_memory: bool = True
    dataloader_pin_memory: bool = True

    # === Distributed training ===
    local_rank: Optional[int] = None
    ddp_find_unused_parameters: bool = False
    deepspeed_config: Optional[str] = None

    # === Seed ===
    seed: int = 42

    # === Logging (WandB, TensorBoard, etc.) ===
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # === Advanced options ===
    gradient_checkpointing: bool = False
    remove_unused_columns: bool = True
    label_smoothing_factor: float = 0.0
    
    # === Scheduler params ===
    lr_scheduler_type: Union[str, LRSchedulerType] = LRSchedulerType.LINEAR
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # === Optimizer ===
    optimizer_type: str = "adamw"  # adamw, adam, sgd, etc.
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # === Misc ===
    resume_from_checkpoint: Optional[str] = None
    disable_tqdm: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    
    # === Method-specific parameters (to be overridden) ===
    # These will be used by specific trainer implementations
    method_specific_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Convert enum to string if needed
        if isinstance(self.lr_scheduler_type, LRSchedulerType):
            self.lr_scheduler_type = self.lr_scheduler_type.value
            
        if isinstance(self.trainer_type, TrainerType):
            self.trainer_type = self.trainer_type.value

        # Validation logic
        if self.warmup_ratio is not None:
            assert 0.0 <= self.warmup_ratio <= 1.0, "warmup_ratio must be between 0 and 1"
            if self.max_steps is not None:
                self.warmup_steps = int(self.warmup_ratio * self.max_steps)

        if self.save_total_limit is not None:
            assert self.save_total_limit > 0, "save_total_limit must be positive"

        assert self.evaluation_strategy in ["no", "steps", "epoch"], "Invalid evaluation_strategy"
        
        # Set run_name if not provided
        if self.run_name is None:
            self.run_name = f"{self.trainer_type}_lr{self.learning_rate}_bs{self.train_batch_size}"
            
        # Ensure metric specification consistency
        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "eval_loss"
            self.greater_is_better = False

    def get_effective_batch_size(self) -> int:
        """Calculate the effective batch size considering gradient accumulation."""
        return self.train_batch_size * self.gradient_accumulation_steps
    
    def get_total_steps(self, num_samples: int) -> int:
        """Calculate total training steps given dataset size."""
        steps_per_epoch = num_samples // self.get_effective_batch_size()
        if self.max_steps is not None:
            return min(self.max_steps, steps_per_epoch * self.num_train_epochs)
        return steps_per_epoch * self.num_train_epochs

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainConfig':
        """Create config from dictionary."""
        return cls(**config_dict)