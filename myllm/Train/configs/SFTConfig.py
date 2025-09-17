from dataclasses import dataclass 
from .BaseConfig import BaseTrainerConfig
from typing import Optional , Dict , Any 

# trainer/configs/sft_config.py
@dataclass
class SFTTrainerConfig(BaseTrainerConfig):
    """Configuration for Supervised Fine-Tuning"""
    max_seq_length: int = 512
    packing: bool = False
    dataset_text_field: str = "text"
    instruction_template: Optional[str] = None
    response_template: Optional[str] = None
    use_peft: bool = False
    peft_config: Optional[Dict[str, Any]] = None
