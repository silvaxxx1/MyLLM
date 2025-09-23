from .base_trainer import BaseTrainer
import logging
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # ✅ Modern AMP API
from typing import Dict, Any
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, MofNCompleteColumn
)
from rich.table import Table
import wandb

# Setup logging and rich console
logger = logging.getLogger(__name__)
console = Console()


class Trainer(BaseTrainer):
    """
    Trainer class for single-device training.
    
    Features:
      - ✅ Modern AMP (Automatic Mixed Precision) using `torch.amp`
      - ✅ Gradient Accumulation for handling large effective batch sizes
      - ✅ Hugging Face-style progress bar with rich
      - ✅ Integrated Weights & Biases (W&B) logging
      - ✅ Automatic evaluation and checkpointing
      - ✅ Optional `torch.compile()` support for runtime graph optimization
    """

    def __init__(self, config, model_config=None, model=None):
        """
        Initialize the trainer with configuration, model config, and optional model.

        Args:
            config: Training configuration (TrainerConfig)
            model_config: Model architecture configuration
            model: Pre-built model instance (optional)
        """
        super().__init__(config, model_config, model)
        self.train_dataloader = None
        self.eval_dataloader = None
        self.latest_eval_loss = None
        self.best_checkpoint_path = None
        self.wandb_url = None

        # ✅ AMP Scaler for automatic mixed precision
        # Using the modern torch.amp API
        self.scaler = GradScaler("cuda")

    # ------------------------------
    # Model and Data Setup
    # ------------------------------
    def setup_model(self) -> nn.Module:
        """
        Build the GPT model and tokenizer.
        Optionally compiles the model for faster training with `torch.compile()`.
        """
        if self.model is None:
            from myllm.model import GPT
            from myllm.Tokenizers.factory import get_tokenizer
            from myllm.Tokenizers.wrapper import TokenizerWrapper

            # Initialize the GPT model
            logger.info(f"Creating model: {self.model_config.name}")
            self.model = GPT(self.model_config)

            # Setup tokenizer
            tokenizer_raw = get_tokenizer(self.config.tokenizer_name)
            self.tokenizer = TokenizerWrapper(tokenizer_raw)

            # Ensure tokenizer has a pad token
            if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token_id", 0)

        # Move model to device (GPU or CPU)
        self.model.to(self.device)

        # ✅ Optional: compile the model for faster runtime
        if getattr(self.config, "use_compile", False):
            logger.info("Compiling model with torch.compile() for optimization")
            self.model = torch.compile(self.model)

        return self.model

    def setup_data(self):
        """Setup datasets and dataloaders. To be implemented by the user."""
        logger.info("Setting up datasets...")
        self.train_dataloader = None
        self.eval_dataloader = None
        logger.info("Datasets setup complete")

    def setup_optimizer(self):
        """Create optimizer and scheduler."""
        if self.model is None:
            raise ValueError("Model must be setup before optimizer")

        # Optimizer selection
        if self.config.optimizer_type.value == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.model_config.learning_rate,
                weight_decay=self.model_config.weight_decay,
                betas=(self.model_config.beta1, self.model_config.beta2)
            )
        else:
            raise NotImplementedError(f"Optimizer {self.config.optimizer_type} not implemented")

        # Optional LR Scheduler
        if self.train_dataloader is not None and self.config.scheduler_type.value == "linear":
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )

        logger.info("Optimizer and scheduler setup complete")

    # ------------------------------
    # Training Step (AMP + Gradient Accumulation)
    # ------------------------------
    def train_step(self, batch) -> Dict[str, Any]:
        """
        Perform a single forward + backward pass with:
          - Mixed precision
          - Gradient accumulation
          - Gradient clipping
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # ✅ Mixed precision forward pass
        with autocast(device_type="cuda"):
            logits = self.model(batch["input_ids"])
            labels = batch.get("labels", batch["input_ids"][:, 1:].contiguous())

            # Ensure logits and labels are the same length
            if logits.size(1) != labels.size(1):
                min_len = min(logits.size(1), labels.size(1))
                logits = logits[:, :min_len].contiguous()
                labels = labels[:, :min_len].contiguous()

            # Cross-entropy loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            # Scale loss if using gradient accumulation
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

        # ✅ Backward pass with AMP scaler
        self.scaler.scale(loss).backward()

        # Gradient clipping for stability
        if self.config.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        # Perform optimizer step after enough accumulation steps
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            self.optimizer.zero_grad()

        return {"loss": loss.item()}

    # ------------------------------
    # Evaluation
    # ------------------------------
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the validation dataset."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0
        num_batches = 0

        # No gradients needed for evaluation
        with torch.no_grad(), autocast(device_type="cuda"):
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                logits = self.model(batch["input_ids"])
                labels = batch.get("labels", batch["input_ids"][:, 1:].contiguous())

                # Align shapes
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

    # ------------------------------
    # Training Loop
    # ------------------------------
    def train(self):
        """
        Main training loop with:
          - AMP
          - Gradient Accumulation
          - Evaluation
          - W&B logging
          - Checkpointing
        """
        if self.train_dataloader is None:
            logger.warning("No training data available")
            return

        # Capture W&B run URL
        self.wandb_url = wandb.run.get_url() if wandb.run else "N/A"

        console.rule(f"[bold green]Starting training for {self.config.num_epochs} epochs")

        # Iterate over epochs
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0
            num_steps = 0

            # Rich progress bar
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

            # Create progress task
            task = progress.add_task(
                f"Epoch {epoch + 1}/{self.config.num_epochs}",
                total=len(self.train_dataloader),
                train_loss=0.0,
                avg_loss=0.0,
                eval_loss=0.0,
                lr=0.0
            )

            # Iterate over batches
            with progress:
                for step, batch in enumerate(self.train_dataloader):
                    step_results = self.train_step(batch)
                    epoch_loss += step_results["loss"]
                    num_steps += 1
                    self.global_step += 1

                    avg_loss = epoch_loss / num_steps
                    lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
                    eval_loss = self.latest_eval_loss or 0.0

                    # Update progress bar
                    progress.update(
                        task,
                        advance=1,
                        train_loss=step_results['loss'],
                        avg_loss=avg_loss,
                        eval_loss=eval_loss,
                        lr=lr
                    )

                    # ✅ Log metrics to W&B
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

                    # ✅ Run evaluation periodically
                    if (
                        self.should_evaluate()
                        and self.eval_dataloader is not None
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        eval_results = self.evaluate()
                        if eval_results:
                            eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
                            self.log_metrics(eval_metrics)
                            console.print(
                                f"[bold magenta]Step {self.global_step} Eval Loss: {eval_results['eval_loss']:.4f}[/bold magenta]"
                            )

                            # Check if this is the best model
                            is_best = self.update_best_metric(eval_results)
                            if is_best:
                                self.best_checkpoint_path = self.save_checkpoint(is_best=True)

                    # ✅ Save checkpoint periodically
                    if self.should_save_checkpoint() and self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

            # End-of-epoch evaluation
            if self.eval_dataloader is not None:
                eval_results = self.evaluate()
                if eval_results:
                    eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
                    eval_metrics["eval/epoch"] = epoch
                    self.log_metrics(eval_metrics)
                    console.print(
                        f"[bold magenta][End of Epoch {epoch + 1}] Eval Loss: {eval_results['eval_loss']:.4f}[/bold magenta]"
                    )

                    # Save best checkpoint
                    is_best = self.update_best_metric(eval_results)
                    if is_best:
                        self.best_checkpoint_path = self.save_checkpoint(is_best=True)

        # ✅ Final summary table
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
