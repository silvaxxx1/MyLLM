from dataclasses import dataclass
from .BaseConfig import BaseTrainerConfig 
from typing import Optional


# trainer/configs/dpo_config.py
@dataclass
class DPOTrainerConfig(BaseTrainerConfig):
    """Configuration for Direct Preference Optimization"""
    beta: float = 0.1
    reference_model_path: Optional[str] = None
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    prompt_field: str = "prompt"
    loss_type: str = "sigmoid"
    reference_free: bool = False
    max_new_tokens: int = 256
    temperature: float = 1.0