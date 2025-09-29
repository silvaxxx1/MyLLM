# trainer/utils/training_flow.py
import wandb
from typing import Dict, Any
import logging
from .progress_utils import update_progress_bar
from .training_utils import ensure_scalar_loss, create_metrics_dict, handle_evaluation_metrics

logger = logging.getLogger(__name__)

class TrainingFlow:
    """
    Standardized training flow that can be reused across trainers
    
    This class encapsulates the common training loop patterns
    while allowing trainers to implement their specific algorithms.
    """
    
    def __init__(self, trainer):
        """
        Initialize training flow with a trainer instance
        
        Args:
            trainer: Trainer instance that implements required methods
        """
        self.trainer = trainer
        
    def run_epoch(self, epoch: int, progress, task) -> tuple:
        """
        Run a single training epoch
        
        Args:
            epoch: Current epoch number
            progress: Rich Progress instance
            task: Progress task ID
            
        Returns:
            tuple: (epoch_loss, num_steps) cumulative loss and step count
        """
        epoch_loss = 0.0
        num_steps = 0
        
        for step, batch in enumerate(self.trainer.train_dataloader):
            # Execute trainer-specific training step
            step_results = self.trainer.training_step(batch)
            
            # Ensure loss is scalar and track it
            loss_value = ensure_scalar_loss(step_results["loss"])
            epoch_loss += loss_value
            num_steps += 1
            self.trainer.global_step += 1

            # Update progress bar with current metrics
            update_progress_bar(
                progress, task, {"loss": loss_value}, epoch_loss, num_steps,
                self.trainer.optimizer, self.trainer.latest_eval_loss
            )

            # Handle logging and evaluation for this step
            self._handle_step_logging(epoch, {"loss": loss_value})
            
        return epoch_loss, num_steps
    
    def _handle_step_logging(self, epoch: int, step_results: Dict[str, Any]):
        """
        Handle logging and evaluation for a single step
        
        Args:
            epoch: Current epoch number
            step_results: Dictionary with step results
        """
        # Log metrics if it's time to log
        if self.trainer.should_log():
            metrics = create_metrics_dict(
                step_results, epoch, self.trainer.global_step,
                self.trainer.optimizer, self.trainer.latest_eval_loss
            )
            self.trainer.log_metrics(metrics)

        # Run evaluation if it's time to evaluate
        if (self.trainer.should_evaluate() and 
            self.trainer.global_step % self.trainer.config.eval_steps == 0):
            self._run_evaluation()
    
    def _run_evaluation(self):
        """Run evaluation and handle results"""
        eval_results = self.trainer.evaluate()
        if eval_results:
            # Convert evaluation results to metrics format
            eval_metrics = handle_evaluation_metrics(
                eval_results, self.trainer.global_step, self.trainer.optimizer
            )
            self.trainer.log_metrics(eval_metrics)
            
            # Print evaluation results to console
            from rich.console import Console
            console = Console()
            console.print(
                f"[bold magenta]Step {self.trainer.global_step} Eval Loss: {eval_results['eval_loss']:.4f}, "
                f"Perplexity: {eval_results.get('perplexity', 0):.2f}[/bold magenta]"
            )
            
            # Check if this is the best model and save if needed
            is_best = self.trainer.update_best_metric(eval_results)
            if is_best:
                self.trainer.best_checkpoint_path = self.trainer.save_checkpoint(is_best=True)
    
    def handle_end_of_epoch(self, epoch: int):
        """
        Handle end-of-epoch evaluation and logging
        
        Args:
            epoch: Completed epoch number
        """
        if self.trainer.eval_dataloader is not None:
            eval_results = self.trainer.evaluate()
            if eval_results:
                # Convert evaluation results to metrics format
                eval_metrics = handle_evaluation_metrics(
                    eval_results, self.trainer.global_step, self.trainer.optimizer
                )
                eval_metrics["eval/epoch"] = epoch
                self.trainer.log_metrics(eval_metrics)
                
                # Print end-of-epoch results to console
                from rich.console import Console
                console = Console()
                console.print(
                    f"[bold magenta][End of Epoch {epoch + 1}] Eval Loss: {eval_results['eval_loss']:.4f}, "
                    f"Perplexity: {eval_results.get('perplexity', 0):.2f}[/bold magenta]"
                )
                
                # Check if this is the best model and save if needed
                is_best = self.trainer.update_best_metric(eval_results)
                if is_best:
                    self.trainer.best_checkpoint_path = self.trainer.save_checkpoint(is_best=True)