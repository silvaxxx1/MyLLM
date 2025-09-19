# engine/accelerators/hf_accelerate_wrapper.py
from accelerate import Accelerator as HFAccelerator
from .base import BaseAccelerator

class HFAccelerateWrapper(BaseAccelerator):
    def setup(self):
        self.acc = HFAccelerator(mixed_precision=self.config.get("mixed_precision", None))

    def prepare_model(self, model):
        # This wrapper expects prepare(model, optimizer, dataloader) in usage
        self.model = model
        return model

    def prepare_optimizer(self, optimizer):
        self.optimizer = optimizer
        return optimizer

    def prepare_all(self, model, optimizer, dataloader):
        self.model, self.optimizer, dataloader = self.acc.prepare(model, optimizer, dataloader)
        return self.model, self.optimizer, dataloader

    def backward(self, loss, optimizer=None):
        self.acc.backward(loss)
