# myllm/Train/factory.py (UPDATED)
from typing import Type
from .base_trainer import BaseTrainer
from .trainer import PretrainTrainer
from .sft_trainer import SFTTrainer
from .sft_classifer import SFTClassifierTrainer  # ADD THIS IMPORT

def create_trainer(trainer_type: str, config, model_config=None, model=None) -> BaseTrainer:
    """
    Factory function to create appropriate trainer
    
    Args:
        trainer_type: Type of trainer ("pretrain", "sft", "sft_classifier")
        config: Trainer configuration object
        model_config: Model configuration object
        model: Optional pre-initialized model
    
    Returns:
        BaseTrainer: Initialized trainer instance
    """
    trainers = {
        "pretrain": PretrainTrainer,
        "sft": SFTTrainer,
        "sft_classifier": SFTClassifierTrainer,  # ADD THIS LINE
        # Add more trainers here as needed
    }
    
    if trainer_type not in trainers:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Available: {list(trainers.keys())}")
    
    return trainers[trainer_type](config, model_config, model)

def create_trainer_from_config(config) -> BaseTrainer:
    """
    Create trainer from config object that has trainer_type attribute
    """
    if not hasattr(config, 'trainer_type'):
        raise ValueError("Config must have 'trainer_type' attribute")
    
    return create_trainer(config.trainer_type, config)