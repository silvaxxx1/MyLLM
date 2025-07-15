# =====================================
# configs/dpo_config.py
"""
Direct Preference Optimization Configuration

Contains DPOConfig class for DPO training parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
import logging
from BaseConfig import BaseTrainConfig

@dataclass
class DPOConfig(BaseTrainConfig):
    """Configuration for Direct Preference Optimization"""
    
    # DPO-specific parameters
    beta: float = 0.1  # DPO temperature parameter
    loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid"
    label_smoothing: float = 0.0
    
    # Reference model
    ref_model_path: Optional[str] = None  # Path to reference model
    ref_model_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Data settings
    max_prompt_length: int = 1024
    max_length: int = 2048
    train_dataset_path: str = ""
    eval_dataset_path: Optional[str] = None
    
    # DPO training specifics
    remove_unused_columns: bool = False
    force_use_ref_model: bool = False
    
    # Model settings
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Advanced DPO settings
    precompute_ref_log_probs: bool = False  # Precompute reference model logprobs
    model_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        # DPO typically needs lower learning rates
        if self.learning_rate > 1e-4:
            logging.warning(f"DPO learning rate {self.learning_rate} might be too high. Consider 1e-6 to 1e-5.")
