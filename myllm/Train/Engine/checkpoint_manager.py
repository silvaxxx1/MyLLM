# engine/checkpoint_manager.py
import torch
import os

class CheckpointManager:
    """Centralized checkpointing with hooks for sharded backends."""

    def __init__(self, model, optimizer=None, scheduler=None, accelerator=None, save_dir="checkpoints"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, step=None, tag="latest"):
        fname = f"checkpoint_{tag}.pt" if step is None else f"checkpoint_{tag}_step{step}.pt"
        path = os.path.join(self.save_dir, fname)
        # Let accelerator handle special checkpointing if available
        acc_state = {}
        if self.accelerator is not None:
            acc_state = getattr(self.accelerator, "state_dict", lambda: {})()
        # Normal save for CPU/GPU single-node or DD P (rank 0 should save)
        # Be careful with FSDP/DeepSpeed: they usually need their own save API
        state = {
            "model_state": self._get_model_state(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "accelerator_state": acc_state,
        }
        torch.save(state, path)
        return path

    def load(self, path, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        self._load_model_state(state.get("model_state", {}))
        if self.optimizer is not None and state.get("optimizer_state"):
            self.optimizer.load_state_dict(state["optimizer_state"])
        if self.scheduler is not None and state.get("scheduler_state"):
            self.scheduler.load_state_dict(state["scheduler_state"])
        return state

    def _get_model_state(self):
        # central hook — supports both nn.Module and model engines
        if hasattr(self.model, "state_dict"):
            return self.model.state_dict()
        # e.g., DeepSpeed engine
        if hasattr(self.model, "module") and hasattr(self.model.module, "state_dict"):
            return self.model.module.state_dict()
        return {}

    def _load_model_state(self, state_dict):
        if not state_dict:
            return
        if hasattr(self.model, "load_state_dict"):
            return self.model.load_state_dict(state_dict)
        if hasattr(self.model, "module") and hasattr(self.model.module, "load_state_dict"):
            return self.model.module.load_state_dict(state_dict)
