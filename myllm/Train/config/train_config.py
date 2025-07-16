# train_config.py

from dataclasses import dataclass, field
from typing import Optional, List, Union

@dataclass
class TrainConfig:
    # === General training parameters ===
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"

    num_train_epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1

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
    fp16: bool = False  # mixed precision training
    bf16: bool = False  # alternative mixed precision, if hardware supports

    # === Logging and checkpointing ===
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: Optional[int] = 3  # limit number of saved checkpoints

    # === Evaluation ===
    evaluation_strategy: str = "steps"  # options: 'no', 'steps', 'epoch'
    eval_delay: int = 0  # number of steps to wait before first eval

    # === Early stopping ===
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None

    # === Data loading ===
    num_workers: int = 4
    pin_memory: bool = True

    # === Distributed training ===
    local_rank: Optional[int] = None  # for distributed training

    # === Seed ===
    seed: int = 42

    # === WandB ===
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # === Advanced options ===
    gradient_checkpointing: bool = False
    max_grad_norm_clip: Optional[float] = None  # alternative max grad norm clipping

    # === Misc ===
    resume_from_checkpoint: Optional[str] = None
    disable_tqdm: bool = False

    # === Model specific parameters (optional override) ===
    model_name_or_path: Optional[str] = None  # for loading pretrained weights
    tokenizer_name_or_path: Optional[str] = None

    # === Scheduler params ===
    lr_scheduler_type: str = "linear"  # linear, cosine, etc.

    def __post_init__(self):
        if self.warmup_ratio is not None:
            assert 0.0 <= self.warmup_ratio <= 1.0, "warmup_ratio must be between 0 and 1"
            if self.max_steps is not None:
                self.warmup_steps = int(self.warmup_ratio * self.max_steps)

        if self.save_total_limit is not None:
            assert self.save_total_limit > 0, "save_total_limit must be positive"

        assert self.evaluation_strategy in ["no", "steps", "epoch"], "Invalid evaluation_strategy"

        # Add more validations if needed
