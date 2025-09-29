# trainer/configs/sft_config.py
from dataclasses import dataclass, field
from typing import Optional, List
from .TrainerConfig import TrainerConfig

@dataclass
class SFTTrainerConfig(TrainerConfig):
    """Configuration for Supervised Fine-Tuning Trainer"""
    
    # ----------------------------
    # SFT Specific Settings
    # ----------------------------
    instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    response_template: str = "### Response:"
    
    # ----------------------------
    # Data Formatting
    # ----------------------------
    train_dataset_path: Optional[str] = None
    eval_dataset_path: Optional[str] = None
    max_examples: Optional[int] = None  # Limit dataset size for quick testing
    
    # ----------------------------
    # Pretrained Weights
    # ----------------------------
    pretrained_variant: Optional[str] = None  # e.g., "gpt2", "gpt2-medium"
    pretrained_path: Optional[str] = None
    
    # ----------------------------
    # SFT-specific Training
    # ----------------------------
    response_only_loss: bool = True  # Only compute loss on response tokens
    instruction_loss_weight: float = 0.0  # Weight for instruction tokens in loss
    
    # ----------------------------
    # Validation
    # ----------------------------
    def validate(self):
        super().validate()
        if not self.instruction_template:
            raise ValueError("instruction_template cannot be empty")
        if not self.response_template:
            raise ValueError("response_template cannot be empty")
        if self.response_only_loss and self.instruction_loss_weight > 0:
            raise ValueError("Cannot use both response_only_loss and instruction_loss_weight")