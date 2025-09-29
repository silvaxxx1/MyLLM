# trainer/pretrain_trainer.py (FIXED with proper WandB setup)
import logging
from typing import Dict, Any
import torch
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table

from .base_trainer import BaseTrainer

import wandb    

logger = logging.getLogger(__name__)
console = Console()

class PretrainTrainer(BaseTrainer):
    """
    Unified pretraining trainer for language modeling
    """

    def __init__(self, config, model_config=None, model=None):
        super().__init__(config, model_config, model)
        
    def setup_model(self) -> torch.nn.Module:
        """Setup model for pretraining"""
        if self.model is None:
            from myllm.model import GPT
            logger.info(f"Creating pretraining model: {self.model_config.name}")
            self.model = GPT(self.model_config)
        
        self.model.to(self.device)
        self.setup_tokenizer()  # Use common tokenizer setup
        
        # Optional compilation
        if getattr(self.config, "use_compile", False):
            try:
                logger.info("Compiling model with torch.compile()")
                self.model = torch.compile(self.model)
            except Exception as e:
                logger.error(f"torch.compile() failed: {e}")
        
        return self.model

    def setup_data(self, train_dataloader=None, eval_dataloader=None):
        """Setup pretraining data"""
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        if eval_dataloader is not None:
            self.eval_dataloader = eval_dataloader
        return self.train_dataloader, self.eval_dataloader

    def _prepare_batch(self, batch):
        """Move batch to device"""
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _get_labels(self, batch):
        """Get labels for pretraining (next token prediction)"""
        return batch.get("labels", batch["input_ids"][:, 1:].contiguous())
    
    def train(self):
        """Training loop for pretraining"""
        if self.train_dataloader is None:
            logger.warning("No training data available")
            return

        # ✅ FIXED: Use the base class setup_wandb method
        self.setup_wandb()
        
        console.rule(f"[bold green]Starting pretraining for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            num_steps = 0

            # Progress bar setup
            progress = self._create_progress_bar(epoch)
            task = progress.add_task(
                f"Epoch {epoch + 1}/{self.config.num_epochs}",
                total=len(self.train_dataloader),
                train_loss=0.0,
                avg_loss=0.0,
                eval_loss=0.0,
                lr=0.0
            )

            with progress:
                for step, batch in enumerate(self.train_dataloader):
                    # Run training step
                    step_results = self.training_step(batch)

                    # Ensure loss is scalar
                    loss_value = step_results["loss"]
                    if isinstance(loss_value, torch.Tensor):
                        loss_value = float(loss_value.item())
                    elif isinstance(loss_value, list):
                        loss_value = sum(float(l.item() if isinstance(l, torch.Tensor) else l) for l in loss_value)

                    # Aggregate epoch loss
                    epoch_loss += loss_value
                    num_steps += 1
                    self.global_step += 1

                    # Update progress
                    self._update_progress(
                        progress, task, {"loss": loss_value}, epoch_loss, num_steps
                    )

                    # Logging and evaluation
                    self._handle_logging_and_evaluation(epoch, {"loss": loss_value})

            # End-of-epoch evaluation
            self._handle_end_of_epoch(epoch)

        # Final summary
        self._print_training_summary()

    def _create_progress_bar(self, epoch):
        """Create rich progress bar"""
        return Progress(
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

    def _update_progress(self, progress, task, step_results, epoch_loss, num_steps):
        """Update progress bar"""
        avg_loss = epoch_loss / max(1, num_steps)
        lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        eval_loss = self.latest_eval_loss or 0.0

        progress.update(
            task,
            advance=1,
            train_loss=step_results['loss'],
            avg_loss=avg_loss,
            eval_loss=eval_loss,
            lr=lr
        )

    def _handle_logging_and_evaluation(self, epoch, step_results):
        """Handle logging and evaluation during training"""
        if self.should_log():
            metrics = {
                "train/loss": step_results["loss"],
                "train/epoch": epoch,
                "train/step": self.global_step,
                "train/learning_rate": self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
            }
            if self.latest_eval_loss is not None:
                metrics["eval/loss"] = self.latest_eval_loss
            
            # ✅ FIXED: Use base class log_metrics which handles WandB
            self.log_metrics(metrics)

        if (self.should_evaluate() and 
            self.global_step % self.config.eval_steps == 0):
            self._run_evaluation()

    def _run_evaluation(self):
        """Run evaluation and save if best"""
        eval_results = self.evaluate()
        if eval_results:
            eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
            # ✅ Add learning rate to eval metrics for better tracking
            eval_metrics["eval/learning_rate"] = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
            self.log_metrics(eval_metrics)
            
            console.print(
                f"[bold magenta]Step {self.global_step} Eval Loss: {eval_results['eval_loss']:.4f}, Perplexity: {eval_results['perplexity']:.2f}[/bold magenta]"
            )
            
            is_best = self.update_best_metric(eval_results)
            if is_best:
                self.best_checkpoint_path = self.save_checkpoint(is_best=True)

    def _handle_end_of_epoch(self, epoch):
        """Handle end of epoch evaluation"""
        if self.eval_dataloader is not None:
            eval_results = self.evaluate()
            if eval_results:
                eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
                eval_metrics["eval/epoch"] = epoch
                eval_metrics["eval/learning_rate"] = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
                self.log_metrics(eval_metrics)
                
                console.print(
                    f"[bold magenta][End of Epoch {epoch + 1}] Eval Loss: {eval_results['eval_loss']:.4f}, Perplexity: {eval_results['perplexity']:.2f}[/bold magenta]"
                )
                
                is_best = self.update_best_metric(eval_results)
                if is_best:
                    self.best_checkpoint_path = self.save_checkpoint(is_best=True)

    def _print_training_summary(self):
        """Print final training summary"""
        table = Table(title="Pretraining Summary", show_lines=True)
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
        console.rule("[bold green]Pretraining completed!")
        
        # ✅ FIXED: Finish WandB run if it exists
        if wandb.run is not None:
            wandb.finish()