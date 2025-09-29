# trainer/configs/sft_config.py
from dataclasses import dataclass, field
from typing import Optional, List, Union
from .TrainerConfig import TrainerConfig

@dataclass
class SFTTrainerConfig(TrainerConfig):
    """Configuration for Supervised Fine-Tuning Trainer - Unified for instruction and classification"""
    
    # ----------------------------
    # Task Type Configuration
    # ----------------------------
    task_type: str = "instruction"  # "instruction" or "classification"
    num_labels: int = 2  # Only used for classification tasks
    
    # ----------------------------
    # SFT Instruction Following Settings
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
    # Classification Specific Settings
    # ----------------------------
    classifier_dropout: float = 0.1  # Dropout for classifier head
    label_names: Optional[List[str]] = None  # Human-readable label names
    
    # ----------------------------
    # Prediction Settings
    # ----------------------------
    max_response_length: int = 100  # For instruction following generation
    temperature: float = 0.7  # For instruction following generation
    top_p: float = 0.9  # For instruction following generation
    
    # ----------------------------
    # Validation
    # ----------------------------
    def validate(self):
        super().validate()
        
        # Validate task type
        if self.task_type not in ["instruction", "classification"]:
            raise ValueError(f"task_type must be 'instruction' or 'classification', got '{self.task_type}'")
        
        # Validate instruction following settings
        if self.task_type == "instruction":
            if not self.instruction_template:
                raise ValueError("instruction_template cannot be empty for instruction task type")
            if not self.response_template:
                raise ValueError("response_template cannot be empty for instruction task type")
            if self.response_only_loss and self.instruction_loss_weight > 0:
                raise ValueError("Cannot use both response_only_loss and instruction_loss_weight")
        
        # Validate classification settings
        if self.task_type == "classification":
            if self.num_labels < 2:
                raise ValueError("num_labels must be at least 2 for classification tasks")
            if self.label_names and len(self.label_names) != self.num_labels:
                raise ValueError(f"label_names length ({len(self.label_names)}) must match num_labels ({self.num_labels})")
            
            # Warn about unused parameters for classification
            if self.response_only_loss:
                print("⚠️  Warning: response_only_loss is not used for classification tasks")
            if self.instruction_loss_weight > 0:
                print("⚠️  Warning: instruction_loss_weight is not used for classification tasks")
        
        # Validate prediction parameters
        if self.temperature <= 0 or self.temperature > 2.0:
            raise ValueError("temperature must be between 0 and 2.0")
        if self.top_p <= 0 or self.top_p > 1.0:
            raise ValueError("top_p must be between 0 and 1.0")
        if self.max_response_length <= 0:
            raise ValueError("max_response_length must be positive")