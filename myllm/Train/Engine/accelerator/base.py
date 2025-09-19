# engine/accelerators/base.py
from abc import ABC, abstractmethod

class BaseAccelerator(ABC):
    """Abstract accelerator interface."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def setup(self):
        """Initialize distributed primitives / process groups."""
        pass

    @abstractmethod
    def prepare_model(self, model):
        """Wrap or move model to device and return wrapped model."""
        return model

    @abstractmethod
    def prepare_optimizer(self, optimizer):
        """Wrap optimizer if needed (eg DeepSpeed)."""
        return optimizer

    @abstractmethod
    def backward(self, loss, optimizer=None):
        """Backprop that may be backend-aware."""
        loss.backward()

    @abstractmethod
    def state_dict(self):
        """Return accelerator state for checkpointing if required."""
        return {}
