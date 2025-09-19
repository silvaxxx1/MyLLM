# engine/trainer_engine.py
from typing import List

class TrainerEngine:
    """
    Generic training engine that works with any Trainer subclass.
    The Trainer must implement:
      - train_step(batch)
      - validation_step(batch)  (optional)
      - on_epoch_end() (optional)
    """

    def __init__(self, trainer, accelerator, optimizer_manager, scheduler_manager=None,
                 checkpoint_manager=None, callbacks: List=None, config=None):
        self.trainer = trainer
        self.accelerator = accelerator
        self.optimizer_manager = optimizer_manager
        self.scheduler_manager = scheduler_manager
        self.checkpoint_manager = checkpoint_manager
        self.callbacks = callbacks or []
        self.config = config or {}

        # built components
        self.optimizer = None
        self.scheduler = None

    def _run_callbacks(self, method_name, **kwargs):
        for cb in self.callbacks:
            getattr(cb, method_name, lambda **k: None)(self, **kwargs)

    def setup(self):
        # accelerator setup
        self.accelerator.setup()
        # model
        self.model = self.trainer.model
        self.model = self.accelerator.prepare_model(self.model)
        # optimizer
        self.optimizer_manager.model = self.model
        self.optimizer = self.optimizer_manager.setup_optimizer()
        self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        # scheduler
        if self.scheduler_manager:
            self.scheduler = self.scheduler_manager.setup_scheduler()
        # trainer hook
        if hasattr(self.trainer, "on_setup"):
            self.trainer.on_setup(self)

    def train(self):
        self._run_callbacks("on_train_start")
        num_epochs = self.config.get("num_epochs", 1)
        for epoch in range(num_epochs):
            self._run_callbacks("on_epoch_start", epoch=epoch)
            self.model.train()
            for step, batch in enumerate(self.trainer.train_dataloader):
                batch = self.trainer.batch_to_device(batch, getattr(self.accelerator, "device", None))
                loss = self.trainer.train_step(batch)
                # backward
                self.accelerator.backward(loss, optimizer=self.optimizer)
                if self.config.get("gradient_clip"):
                    from .utils import EngineUtils
                    EngineUtils.clip_gradients(self.model, self.config["gradient_clip"])
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._run_callbacks("on_step_end", step=step, loss=loss)
            # epoch end
            if hasattr(self.trainer, "on_epoch_end"):
                metrics = self.trainer.on_epoch_end(epoch)
            else:
                metrics = None
            self._run_callbacks("on_epoch_end", epoch=epoch, metrics=metrics)
            # optional checkpoint
            if self.checkpoint_manager and self.config.get("save_every_epoch", False):
                self.checkpoint_manager.save(step=None, tag=f"epoch{epoch}")
        self._run_callbacks("on_train_end")

    def evaluate(self):
        if hasattr(self.trainer, "evaluate"):
            return self.trainer.evaluate(self)
        return None
