# engine/optimizer.py
import torch

class OptimizerManager:
    """Create optimizer (and optionally instantiate DeepSpeed/ZeRO-related wrappers)."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = None

    def setup_optimizer(self):
        opt_cfg = self.config.get("optimizer", {})
        name = opt_cfg.get("name", "adamw").lower()
        lr = opt_cfg.get("lr", 5e-5)
        weight_decay = opt_cfg.get("weight_decay", 0.0)

        if name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=opt_cfg.get("momentum", 0.9))
        else:
            raise ValueError(f"Unknown optimizer {name}")
        return self.optimizer
