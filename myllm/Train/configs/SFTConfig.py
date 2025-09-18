# trainer/configs/sft_config.py
from dataclasses import dataclass
from .TrainerConfig import TrainerConfig
from typing import Optional, Dict, Any

@dataclass
class SFTTrainerConfig(TrainerConfig):
    """Supervised Fine-Tuning configuration"""
    packing: bool = False
    dataset_text_field: str = "text"
    instruction_template: Optional[str] = None
    response_template: Optional[str] = None
    use_peft: bool = False
    peft_config: Optional[Dict[str, Any]] = None