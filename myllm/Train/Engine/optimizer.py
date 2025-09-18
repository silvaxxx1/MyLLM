# engine/optimizer.py
import torch

class OptimizerManager:
    """Handles optimizer creation and configuration"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = None

    def setup_optimizer(self):
        # Placeholder for optimizer selection
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=getattr(self.config, "learning_rate", 5e-5),
            weight_decay=getattr(self.config, "weight_decay", 0.0)
        )
        return self.optimizer
