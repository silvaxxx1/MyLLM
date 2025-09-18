
# trainer/configs/dpo_config.py
from dataclasses import dataclass
from .TrainerConfig import TrainerConfig
from typing import Optional

@dataclass
class DPOTrainerConfig(TrainerConfig):
    """Direct Preference Optimization configuration"""
    beta: float = 0.1
    reference_model_path: Optional[str] = None
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    prompt_field: str = "prompt"