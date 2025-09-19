# engine/accelerators/fsdp_accelerator.py
# Minimal FSDP wrapper (requires torch >= x.y)
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from .base import BaseAccelerator

class FSDPAccelerator(BaseAccelerator):
    def setup(self):
        # assumes dist is already initialized externally via ddp_accelerator pattern
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_model(self, model):
        model.to(self.device)
        fsdp_model = FSDP(model)
        return fsdp_model

    def prepare_optimizer(self, optimizer):
        # optimizer works normally with FSDP; ZeRO-like behavior handled internally
        return optimizer

    def backward(self, loss, optimizer=None):
        loss.backward()

    def state_dict(self):
        return {}
