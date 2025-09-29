# trainer/__init__.py
"""
ML Training Framework integrated with existing project architecture
Compatible with model.py, ModelConfig, and api.py
"""

from .configs import (
    TrainerConfig, SFTTrainerConfig, PPOTrainerConfig, DPOTrainerConfig,
    LoggingBackend, OptimizerType, SchedulerType
)
from .base_trainer import BaseTrainer
from .trainer import PretrainTrainer
from .sft_trainer import SFTTrainer
from .ppo_trainer import PPOTrainer
from .dpo_trainer import DPOTrainer
from .utils import ConfigManager, LoggingManager

__version__ = "1.0.0"