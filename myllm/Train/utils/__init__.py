# trainer/utils/__init__.py
"""
Training utilities package

This package contains reusable utilities for training loops,
progress tracking, metrics handling, and model management.
"""

from .config_manager import ConfigManager
from .logging_utils import LoggingManager
from .progress_utils import create_progress_bar, update_progress_bar
from .training_utils import ensure_scalar_loss, create_metrics_dict, handle_evaluation_metrics
from .summary_utils import create_training_summary_table, print_training_completion
from .model_utils import setup_model_compilation, load_pretrained_weights
from .training_flow import TrainingFlow
from .memory_utils import cleanup_memory

__all__ = [
    # Progress utilities
    'create_progress_bar',
    'update_progress_bar',
    
    # Training utilities
    'ensure_scalar_loss', 
    'create_metrics_dict',
    'handle_evaluation_metrics',
    
    # Summary utilities
    'create_training_summary_table',
    'print_training_completion',
    
    # Model utilities
    'setup_model_compilation',
    'load_pretrained_weights',
    
    # Training flow
    'TrainingFlow',
    
    # Memory management
    'cleanup_memory',

    "ConfigManager",
    "LoggingManager"
]