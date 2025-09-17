from dataclasses import dataclass 
from .BaseConfig import BaseTrainerConfig 
from typing import Optional 

# trainer/configs/ppo_config.py
@dataclass
class PPOTrainerConfig(BaseTrainerConfig):
    """Configuration for PPO Training"""
    ppo_epochs: int = 4
    mini_batch_size: int = 1
    reward_model_path: Optional[str] = None
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    adap_kl_ctrl: bool = True
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9