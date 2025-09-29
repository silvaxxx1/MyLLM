# trainer/sft_trainer.py (FIXED with proper WandB setup)
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

class SFTTrainer(BaseTrainer):
    """
    Unified SFT trainer for instruction following
    """

    def __init__(self, config, model_config=None, model=None):
        super().__init__(config, model_config, model)
        
        self.instruction_template = getattr(config, 'instruction_template', 
                                          "### Instruction:\n{instruction}\n\n### Response:\n{response}")
        self.response_marker = getattr(config, 'response_template', "### Response:")

    def setup_model(self) -> torch.nn.Module:
        """Setup model for SFT with optional pretrained weights"""
        from myllm.api import LLM

        if self.model is not None:
            logger.info("Using externally provided model")
            self.model.to(self.device)
            self.setup_tokenizer()  # ✅ Setup tokenizer even with external model
            return self.model

        if self.model_config is None:
            raise ValueError("model_config must be provided to create a model")

        logger.info(f"Creating LLM: {self.model_config.name} on {self.device}")
        self.llm = LLM(config=self.model_config, device=self.device)

        # Load pretrained weights if specified
        pretrained_variant = getattr(self.config, "pretrained_variant", None)
        if pretrained_variant:
            logger.info(f"Loading pretrained weights: {pretrained_variant}")
            self.llm.load(model_variant=pretrained_variant, model_family="gpt2")

        self.model = self.llm.model
        self.setup_tokenizer()  # Use common tokenizer setup
        
        self.model.to(self.device)

        # Optional compilation
        if getattr(self.config, "use_compile", False):
            try:
                logger.info("Compiling model with torch.compile()")
                self.model = torch.compile(self.model)
            except Exception as e:
                logger.error(f"torch.compile() failed: {e}")

        return self.model

    def setup_data(self, train_dataloader=None, eval_dataloader=None):
        """Setup SFT data"""
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        if eval_dataloader is not None:
            self.eval_dataloader = eval_dataloader
        return self.train_dataloader, self.eval_dataloader

    def _prepare_batch(self, batch):
        """Move batch to device"""
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _get_labels(self, batch):
        """Get labels for SFT with response masking"""
        return batch.get("labels", self._create_response_mask(
            batch["input_ids"], 
            batch.get("attention_mask"), 
            batch.get("instruction", "")
        ))

    def _create_response_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, instruction_text: str) -> torch.Tensor:
        """Create response mask for SFT training"""
        labels = input_ids.clone()
        
        for i in range(input_ids.size(0)):
            text = self.tokenizer.decode(input_ids[i], skip_special_tokens=False)
            response_start = text.find(self.response_marker)
            
            if response_start != -1:
                prefix = text[:response_start + len(self.response_marker)]
                prefix_tokens = self.tokenizer.encode(prefix)
                mask_until = len(prefix_tokens)
                if mask_until < labels.size(1):
                    labels[i, :mask_until] = -100
        
        # Apply attention mask
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)
            
        return labels

    def train(self):
        """Training loop for SFT"""
        if self.train_dataloader is None:
            logger.warning("No training data available")
            return

        # ✅ FIXED: Use the base class setup_wandb method
        self.setup_wandb()
        
        console.rule(f"[bold green]Starting SFT Training for {self.config.num_epochs} epochs")

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
                    step_results = self.training_step(batch)
                    epoch_loss += step_results["loss"]
                    num_steps += 1
                    self.global_step += 1

                    # Update progress
                    self._update_progress(
                        progress, task, step_results, epoch_loss, num_steps
                    )

                    # Logging and evaluation
                    self._handle_logging_and_evaluation(epoch, step_results)

            # End-of-epoch evaluation
            self._handle_end_of_epoch(epoch)

        # Final summary
        self._print_training_summary()

    def _create_progress_bar(self, epoch):
        """Create rich progress bar for SFT"""
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
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
        eval_loss = self.latest_eval_loss or 0.0

        progress.update(
            task, 
            advance=1, 
            train_loss=step_results["loss"], 
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
            
            # ✅ FIXED: Use base class log_metrics
            self.log_metrics(metrics)

        if (self.should_evaluate() and 
            self.global_step % self.config.eval_steps == 0):
            self._run_evaluation()

    def _run_evaluation(self):
        """Run evaluation for SFT"""
        eval_results = self.evaluate()
        if eval_results:
            eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
            # ✅ Add learning rate to eval metrics
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
        """Print SFT training summary"""
        table = Table(title="SFT Training Summary", show_lines=True)
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
        console.rule("[bold green]SFT Training Completed!")
        
        # ✅ FIXED: Finish WandB run if it exists
        if wandb.run is not None:
            wandb.finish()