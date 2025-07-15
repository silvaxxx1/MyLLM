# configs/__init__.py
"""
MyLLM Configuration Package

This package provides configuration classes for all aspects of LLM training:
- BaseTrainConfig: Core training parameters
- SFTConfig: Supervised Fine-Tuning  
- DPOConfig: Direct Preference Optimization
- PPOConfig: Proximal Policy Optimization
- TrainConfig: Unified training configuration
- ModelConfig: Model architecture configuration
- GenerationConfig: Text generation configuration

Usage:
    from configs import SFTConfig, ModelConfig
    from configs.unified_config import TrainConfig
    
    # Or import everything
    from configs import *
"""

from BaseConfig import BaseTrainConfig
from SFTConfig import SFTConfig
from DPOConfig import DPOConfig
from PPOConfig import PPOConfig
from  UniConfig import TrainConfig, create_sft_config, create_dpo_config, create_ppo_config
from ModelConfig import ModelConfig

__all__ = [
    'BaseTrainConfig',
    'SFTConfig', 
    'DPOConfig',
    'PPOConfig',
    'TrainConfig',
    'ModelConfig',
    'GenerationConfig',
    'create_sft_config',
    'create_dpo_config', 
    'create_ppo_config'
]

# Version info
__version__ = "1.0.0"

# Common utilities
RECOMMENDED_CONFIGS = {
    "sft_small": {
        "learning_rate": 5e-5,
        "batch_size": 8,
        "max_epochs": 3,
        "warmup_ratio": 0.1,
        "scheduler": "cosine"
    },
    "sft_large": {
        "learning_rate": 1e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "max_epochs": 1,
        "warmup_ratio": 0.03,
        "scheduler": "linear"
    },
    "dpo_default": {
        "learning_rate": 1e-6,
        "batch_size": 4,
        "max_epochs": 1,
        "beta": 0.1,
        "warmup_ratio": 0.1
    },
    "ppo_default": {
        "learning_rate": 1e-5,
        "batch_size": 64,
        "mini_batch_size": 16,
        "ppo_epochs": 4,
        "clip_range": 0.2,
        "target_kl": 0.01
    }
}

def get_recommended_config(training_type: str, model_size: str = "default") -> dict:
    """Get recommended configuration for specific training type and model size"""
    key = f"{training_type}_{model_size}"
    if key in RECOMMENDED_CONFIGS:
        return RECOMMENDED_CONFIGS[key]
    elif f"{training_type}_default" in RECOMMENDED_CONFIGS:
        return RECOMMENDED_CONFIGS[f"{training_type}_default"]
    else:
        return {}



# =====================================

