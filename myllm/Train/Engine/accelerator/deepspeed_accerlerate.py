# engine/accelerators/deepspeed_accelerator.py
# This is the minimal DeepSpeed wrapper. Real use requires deepspeed package.
import deepspeed
from .base import BaseAccelerator

class DeepSpeedAccelerator(BaseAccelerator):
    def __init__(self, config):
        super().__init__(config)
        self.engine = None
        self.ds_config = config.get("deepspeed_config", {})

    def setup(self):
        # DeepSpeed expects args e.g. --local_rank, but we assume launcher handled it
        pass

    def prepare_model(self, model):
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=self.ds_config
        )
        self.engine = model_engine
        return model_engine

    def prepare_optimizer(self, optimizer):
        # deepspeed manages optimizer when used via initialize
        return self.engine.optimizer if self.engine is not None else optimizer

    def backward(self, loss, optimizer=None):
        # deepspeed engine handles backward
        self.engine.backward(loss)

    def state_dict(self):
        # deepspeed saves checkpoints via engine.save_checkpoint
        return {}
