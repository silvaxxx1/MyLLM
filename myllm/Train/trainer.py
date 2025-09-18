from .base_trainer import BaseTrainer
import logging
import torch
import torch.nn as nn
from typing import Dict, Any
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
import wandb

logger = logging.getLogger(__name__)
console = Console()

class Trainer(BaseTrainer):
    """
    Trainer with Hugging Face-style progress bar including colored metrics
    using rich, along with W&B logging and a polished training summary table.
    """

    def __init__(self, config, model_config=None, model=None):
        super().__init__(config, model_config, model)
        self.train_dataloader = None
        self.eval_dataloader = None
        self.latest_eval_loss = None
        self.best_checkpoint_path = None
        self.wandb_url = None

    def setup_model(self) -> nn.Module:
        if self.model is None:
            from myllm.model import GPT
            from myllm.Tokenizers.factory import get_tokenizer
            from myllm.Tokenizers.wrapper import TokenizerWrapper

            logger.info(f"Creating model: {self.model_config.name}")
            self.model = GPT(self.model_config)

            tokenizer_raw = get_tokenizer(self.config.tokenizer_name)
            self.tokenizer = TokenizerWrapper(tokenizer_raw)

            if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token_id", 0)

        return self.model

    def setup_data(self):
        logger.info("Setting up datasets...")
        self.train_dataloader = None
        self.eval_dataloader = None
        logger.info("Datasets setup complete")

    def setup_optimizer(self):
        if self.model is None:
            raise ValueError("Model must be setup before optimizer")

        if self.config.optimizer_type.value == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.model_config.learning_rate,
                weight_decay=self.model_config.weight_decay,
                betas=(self.model_config.beta1, self.model_config.beta2)
            )
        else:
            raise NotImplementedError(f"Optimizer {self.config.optimizer_type} not implemented")

        if self.train_dataloader is not None and self.config.scheduler_type.value == "linear":
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )

        logger.info("Optimizer and scheduler setup complete")

    def train_step(self, batch) -> Dict[str, Any]:
        self.model.train()
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        logits = self.model(batch["input_ids"])
        labels = batch.get("labels", batch["input_ids"][:, 1:].contiguous())

        if logits.size(1) != labels.size(1):
            min_len = min(logits.size(1), labels.size(1))
            logits = logits[:, :min_len].contiguous()
            labels = labels[:, :min_len].contiguous()

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        loss.backward()

        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

        return {"loss": loss.item()}

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                logits = self.model(batch["input_ids"])
                labels = batch.get("labels", batch["input_ids"][:, 1:].contiguous())

                if logits.size(1) != labels.size(1):
                    min_len = min(logits.size(1), labels.size(1))
                    logits = logits[:, :min_len].contiguous()
                    labels = labels[:, :min_len].contiguous()

                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.latest_eval_loss = avg_loss
        return {"eval_loss": avg_loss}

    def _train_loop(self):
        """Full training loop with Hugging Face-style table and colored metrics"""
        if self.train_dataloader is None:
            logger.warning("No training data available")
            return

        # Capture W&B run URL
        if wandb.run:
            self.wandb_url = wandb.run.get_url()
        else:
            self.wandb_url = "N/A"

        console.rule(f"[bold green]Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0
            num_steps = 0

            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[yellow]LR: {task.fields[lr]:.2e}"),
                TextColumn("[red]Train Loss: {task.fields[train_loss]:.4f}"),
                TextColumn("[cyan]Avg Loss: {task.fields[avg_loss]:.4f}"),
                TextColumn("[magenta]Eval Loss: {task.fields[eval_loss]:.4f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            )

            task = progress.add_task(
                f"Epoch {epoch+1}/{self.config.num_epochs}", total=len(self.train_dataloader),
                train_loss=0.0, avg_loss=0.0, eval_loss=0.0, lr=0.0
            )

            with progress:
                for step, batch in enumerate(self.train_dataloader):
                    step_results = self.train_step(batch)
                    epoch_loss += step_results["loss"]
                    num_steps += 1
                    self.global_step += 1

                    avg_loss = epoch_loss / num_steps
                    lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
                    eval_loss = self.latest_eval_loss or 0.0

                    progress.update(task, advance=1, train_loss=step_results['loss'],
                                    avg_loss=avg_loss, eval_loss=eval_loss, lr=lr)

                    # W&B logging
                    if self.should_log():
                        metrics = {
                            "train/loss": step_results["loss"],
                            "train/epoch_avg_loss": avg_loss,
                            "train/epoch": epoch,
                            "train/step": self.global_step
                        }
                        if self.latest_eval_loss is not None:
                            metrics["eval/loss"] = self.latest_eval_loss
                        self.log_metrics(metrics)

                    # Evaluation during training
                    if self.should_evaluate() and self.eval_dataloader is not None and self.global_step % self.config.eval_steps == 0:
                        eval_results = self.evaluate()
                        if eval_results:
                            eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
                            self.log_metrics(eval_metrics)
                            console.print(f"[bold magenta]Step {self.global_step} Eval Loss: {eval_results['eval_loss']:.4f}[/bold magenta]")
                            is_best = self.update_best_metric(eval_results)
                            if is_best:
                                self.best_checkpoint_path = self.save_checkpoint(is_best=True)

                    # Checkpoint saving
                    if self.should_save_checkpoint() and self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

            # End-of-epoch evaluation
            if self.eval_dataloader is not None:
                eval_results = self.evaluate()
                if eval_results:
                    eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
                    eval_metrics["eval/epoch"] = epoch
                    self.log_metrics(eval_metrics)
                    console.print(f"[bold magenta][End of Epoch {epoch+1}] Eval Loss: {eval_results['eval_loss']:.4f}[/bold magenta]")
                    is_best = self.update_best_metric(eval_results)
                    if is_best:
                        self.best_checkpoint_path = self.save_checkpoint(is_best=True)

        # Final training summary table
        table = Table(title="Training Summary", show_lines=True)
        table.add_column("Epochs", justify="center")
        table.add_column("Final Eval Loss", justify="center")
        table.add_column("Best Checkpoint", justify="center")
        table.add_column("W&B Run URL", justify="center")
        table.add_row(
            str(self.config.num_epochs),
            f"{self.latest_eval_loss:.4f}" if self.latest_eval_loss else "N/A",
            self.best_checkpoint_path if self.best_checkpoint_path else "N/A",
            self.wandb_url
        )
        console.print(table)
        console.rule("[bold green]Training completed!")
