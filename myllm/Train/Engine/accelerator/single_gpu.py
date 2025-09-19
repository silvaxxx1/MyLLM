# engine/accelerators/single_gpu.py
import torch
from .base import BaseAccelerator

class SingleGPUAccelerator(BaseAccelerator):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_model(self, model):
        model.to(self.device)
        return model

    def prepare_optimizer(self, optimizer):
        return optimizer

    def backward(self, loss, optimizer=None):
        loss.backward()

    def state_dict(self):
        return {"device": str(self.device)}
