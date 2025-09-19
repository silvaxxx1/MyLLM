# engine/lr_scheduler.py
from torch.optim.lr_scheduler import LambdaLR

class SchedulerManager:
    """Wrap LR scheduler choices."""

    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config
        self.scheduler = None

    def setup_scheduler(self):
        sched_cfg = self.config.get("scheduler", {})
        name = sched_cfg.get("name", None)
        if name == "linear_warmup":
            total_steps = sched_cfg["total_steps"]
            warmup_steps = sched_cfg.get("warmup_steps", int(0.1 * total_steps))
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = None
        return self.scheduler
