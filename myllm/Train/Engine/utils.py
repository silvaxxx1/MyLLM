# engine/utils.py
import torch
from typing import Any, Dict

class EngineUtils:
    @staticmethod
    def clip_gradients(model, max_norm):
        parameters = [p for p in model.parameters() if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    @staticmethod
    def batch_to_device(batch: Dict[str, Any], device):
        def _move(x):
            return x.to(device) if hasattr(x, "to") else x
        if isinstance(batch, dict):
            return {k: _move(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(_move(x) for x in batch)
        return _move(batch)
