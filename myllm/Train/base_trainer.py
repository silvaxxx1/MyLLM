import os
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from tqdm import tqdm
from accelerate import Accelerator
import wandb

from config.train_config import TrainConfig
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import init_wandb, log_metrics


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        eval_loader,
        config: TrainConfig,
        resume_from_checkpoint: str = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config

        self.accelerator = Accelerator(
            mixed_precision="fp16" if config.fp16 else ("bf16" if config.bf16 else "no"),
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb" if config.use_wandb else None,
        )

        self.optimizer = self.configure_optimizer()

        self.model, self.optimizer, self.train_loader, self.eval_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.eval_loader
        )

        self.scheduler = self.configure_scheduler()
        self.global_step = 0

        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logging_dir, exist_ok=True)

        if self.config.use_wandb and self.accelerator.is_main_process:
            init_wandb(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                run_name=self.config.wandb_run_name,
                config=self.config.__dict__
            )

        if resume_from_checkpoint:
            load_checkpoint(self, resume_from_checkpoint)

    def configure_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    def configure_scheduler(self):
        total_steps = self.config.max_steps or (len(self.train_loader) * self.config.num_train_epochs)
        if self.config.lr_scheduler_type == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0 if self.config.warmup_steps == 0 else 0.0,
                total_iters=total_steps
            )
        else:
            # Add other schedulers here as needed
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=total_steps // 3, gamma=0.1)

    @abstractmethod
    def compute_loss(self, batch) -> torch.Tensor:
        """Implement loss calculation in subclasses"""
        pass

    def train(self):
        self.model.train()
        for epoch in range(self.config.num_train_epochs):
            epoch_loss = 0.0
            progress = tqdm(
                self.train_loader,
                disable=self.config.disable_tqdm or not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch + 1}/{self.config.num_train_epochs}"
            )

            for step, batch in enumerate(progress):
                loss = self.compute_loss(batch) / self.config.gradient_accumulation_steps

                self.accelerator.backward(loss)
                epoch_loss += loss.item()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.config.logging_steps == 0 and self.accelerator.is_local_main_process:
                        avg_loss = epoch_loss / (step + 1)
                        print(f"[Step {self.global_step}] Loss: {avg_loss:.4f}")
                        if self.config.use_wandb:
                            log_metrics({"train_loss": avg_loss, "step": self.global_step})

                    if self.global_step % self.config.eval_steps == 0:
                        self.evaluate()

                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

                    if self.config.max_steps and self.global_step >= self.config.max_steps:
                        print("Max steps reached. Ending training.")
                        return

    def evaluate(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(
                self.eval_loader,
                disable=self.config.disable_tqdm or not self.accelerator.is_local_main_process,
                desc="Evaluating"
            ):
                loss = self.compute_loss(batch)
                losses.append(loss.item())
        avg_loss = sum(losses) / len(losses) if losses else float('nan')

        if self.accelerator.is_local_main_process:
            print(f"[Eval] Loss: {avg_loss:.4f}")
            if self.config.use_wandb:
                log_metrics({"eval_loss": avg_loss, "step": self.global_step})

        self.model.train()

    def save_checkpoint(self, step=None):
        if not self.accelerator.is_local_main_process:
            return
        step = step or self.global_step
        save_checkpoint(self, step)

    def load_checkpoint(self, path):
        load_checkpoint(self, path)
