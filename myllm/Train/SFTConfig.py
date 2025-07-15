# configs/sft_config.py
"""
Supervised Fine-Tuning Configuration

Contains SFTConfig class for supervised fine-tuning parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from BaseConfig import BaseTrainConfig

@dataclass
class SFTConfig(BaseTrainConfig):
    """Configuration for Supervised Fine-Tuning"""
    
    # SFT-specific parameters
    packing: bool = False  # Pack multiple sequences into one
    dataset_text_field: str = "text"  # Field name in dataset containing text
    max_seq_length: int = 2048
    
    # Data formatting
    formatting_func: Optional[str] = None  # Function name for custom formatting
    response_template: Optional[str] = None  # Template for responses
    instruction_template: Optional[str] = None  # Template for instructions
    
    # Loss settings
    neftune_noise_alpha: Optional[float] = None  # NEFTune noise injection
    
    # Dataset settings
    train_dataset_path: str = ""
    eval_dataset_path: Optional[str] = None
    dataset_num_proc: int = 4
    
    # Model specific
    use_lora: bool = False  # Use LoRA for efficient fine-tuning
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    def __post_init__(self):
        super().__post_init__()
        if self.max_length != self.max_seq_length:
            self.max_length = self.max_seq_length
