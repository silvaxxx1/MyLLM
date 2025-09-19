# engine/accelerators/ddp_accelerator.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .base import BaseAccelerator
import os

class DDPAccelerator(BaseAccelerator):
    def setup(self):
        # Minimal launch/setup: expects torchrun or equivalent
        if not dist.is_available():
            raise RuntimeError("torch.distributed not available")
        if not dist.is_initialized():
            dist.init_process_group(backend=self.config.get("backend", "nccl"))
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        torch.cuda.set_device(self.config.get("local_rank", int(os.environ.get("LOCAL_RANK", 0))))
        self.device = torch.device("cuda", torch.cuda.current_device())

    def prepare_model(self, model):
        model.to(self.device)
        model = DDP(model, device_ids=[self.device.index])
        return model

    def prepare_optimizer(self, optimizer):
        return optimizer

    def backward(self, loss, optimizer=None):
        loss.backward()

    def state_dict(self):
        return {"rank": dist.get_rank(), "world_size": dist.get_world_size()}
