from .base_trainer import BaseTrainer 
from typing import Dict, Any, Optional 

import torch.nn as nn 

# trainer/sft_trainer.py
class SFTTrainer(BaseTrainer):
    """Supervised Fine-Tuning Trainer"""
    
    def __init__(self, config, model: Optional[nn.Module] = None):
        super().__init__(config, model)
        # SFT-specific initialization
    
    def setup_model(self) -> nn.Module:
        # TODO: Implement SFT model setup with instruction formatting
        pass
    
    def setup_data(self):
        # TODO: Implement SFT data setup with instruction/response formatting
        pass
    
    def setup_optimizer(self):
        # TODO: Implement SFT optimizer setup (similar to base with PEFT support)
        pass
    
    def train_step(self, batch) -> Dict[str, Any]:
        # TODO: Implement SFT training step with instruction masking
        pass
    
    def evaluate(self) -> Dict[str, float]:
        # TODO: Implement SFT evaluation with generation metrics
        pass
    
    def _train_loop(self):
        # TODO: Implement SFT training loop
        pass