from dataclasses import dataclass
from .BaseConfig import BaseTrainerConfig
from typing import Optional 


# trainer/configs/trainer_config.py
@dataclass
class TrainerConfig(BaseTrainerConfig):
    """Configuration for pre-training"""
    max_seq_length: int = 1024
    mlm_probability: float = 0.15
    block_size: int = 1024
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    resume_from_checkpoint: Optional[str] = None