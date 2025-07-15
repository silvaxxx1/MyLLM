
# =====================================
# configs/ppo_config.py
"""
Proximal Policy Optimization Configuration

Contains PPOConfig class for PPO training parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from BaseConfig import BaseTrainConfig

@dataclass
class PPOConfig(BaseTrainConfig):
    """Configuration for Proximal Policy Optimization"""
    
    # PPO-specific parameters
    ppo_epochs: int = 4
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    target_kl: float = 0.01
    
    # Value function and advantage estimation
    vf_coef: float = 0.1  # Value function coefficient
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    use_score_scaling: bool = False
    use_score_norm: bool = False
    
    # Reward model
    reward_model_path: Optional[str] = None
    reward_model_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Generation settings for PPO
    generation_config: Optional[Dict[str, Any]] = None
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    
    # Experience buffer
    batch_size: int = 64  # PPO batch size
    mini_batch_size: int = 16  # Mini-batch size for PPO updates
    gradient_accumulation_steps: int = 1
    
    # Data settings
    query_dataset_path: str = ""
    max_prompt_length: int = 1024
    
    # Model settings
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Advanced PPO settings
    entropy_coef: float = 0.01  # Entropy coefficient for exploration
    max_grad_norm: float = 1.0
    optimize_cuda_cache: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        if self.clip_range_vf is None:
            self.clip_range_vf = self.clip_range
        
        # Ensure mini_batch_size divides batch_size
        if self.batch_size % self.mini_batch_size != 0:
            raise ValueError(f"batch_size ({self.batch_size}) must be divisible by mini_batch_size ({self.mini_batch_size})")
