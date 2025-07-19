# ppo_config.py 
from .BaseConfig import TrainConfig, TrainerType
from dataclasses import dataclass 
from typing import Optional


@dataclass 
class DPOConfig(TrainConfig):
    """Direct Preference Optimization specific configuration."""
    
    trainer_type: TrainerType = TrainerType.DPO
    
    # DPO hyperparameters
    beta: float = 0.1  # Regularization parameter
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo, kto_pair
    reference_free: bool = False  # Whether to use reference model
    
    # Data fields
    prompt_field: str = "prompt"
    chosen_field: str = "chosen" 
    rejected_field: str = "rejected"
    
    # Model configuration
    reference_model_name: Optional[str] = None  # Path to reference model
    reference_model_adapter_name: Optional[str] = None
    
    # Advanced DPO options
    label_smoothing: float = 0.0
    simpo_gamma: float = 1.0  # For SimPO variant
    cpo_alpha: float = 1.0  # For CPO variant
    
    # Regularization
    sft_weight: float = 0.0  # Weight for SFT loss term
    
    def __post_init__(self):
        super().__post_init__()
        # DPO typically uses lower learning rates
        if self.learning_rate == 5e-5:  # Default value
            self.learning_rate = 5e-7
