# trainer/configs/ppo_config.py
from dataclasses import dataclass
from .TrainerConfig import TrainerConfig
from typing import Optional

@dataclass
class PPOTrainerConfig(TrainerConfig):
    """PPO Training configuration"""
    ppo_epochs: int = 4
    mini_batch_size: int = 1
    reward_model_path: Optional[str] = None
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    gamma: float = 1.0
    max_new_tokens: int = 256
    temperature: float = 1.0