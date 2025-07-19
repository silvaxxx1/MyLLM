# sft_config.py 

# specific_configs.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from myllm.Train.config.BaseConfig import TrainConfig, TrainerType

@dataclass
class SFTConfig(TrainConfig):
    """Supervised Fine-tuning specific configuration."""
    
    trainer_type: TrainerType = TrainerType.SFT
    
    # SFT specific parameters
    packing: bool = False  # Pack multiple sequences into single batch
    dataset_text_field: str = "text"
    max_seq_length: Optional[int] = None
    
    # Data formatting
    formatting_func: Optional[str] = None  # Name of formatting function
    response_template: Optional[str] = None  # Template for response in chat format
    instruction_template: Optional[str] = None
    
    # Loss computation
    ignore_index: int = -100  # Token index to ignore in loss calculation



@dataclass
class PPOConfig(TrainConfig):
    """Proximal Policy Optimization specific configuration."""
    
    trainer_type: TrainerType = TrainerType.PPO
    
    # PPO hyperparameters
    ppo_epochs: int = 4
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    adap_kl_ctrl: bool = True
    
    # Value function
    use_value_head: bool = True
    vf_coef: float = 0.1  # Value function coefficient
    
    # Policy gradients
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1.0  # Discount factor
    lam: float = 0.95  # GAE lambda
    
    # Rollout parameters
    rollout_batch_size: int = 512
    step_size: int = 256
    forward_batch_size: int = 16
    
    # Reward model
    reward_model_name: Optional[str] = None
    reward_baseline: float = 0.0
    normalize_rewards: bool = True
    
    # Generation parameters for rollouts
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Advanced PPO options
    whiten_rewards: bool = True
    ratio_threshold: float = 10.0  # Clip ratios beyond this threshold
    
    def __post_init__(self):
        super().__post_init__()
        # PPO typically needs specific batch size relationships
        assert self.rollout_batch_size >= self.train_batch_size * self.gradient_accumulation_steps

