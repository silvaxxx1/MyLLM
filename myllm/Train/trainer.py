# trainer/pretrain_trainer.py (REFACTORED)
import logging
from typing import Dict, Any
import torch

from .base_trainer import BaseTrainer
from .utils.progress_utils import create_progress_bar
from .utils.training_flow import TrainingFlow
from .utils.summary_utils import create_training_summary_table, print_training_completion
from .utils.model_utils import setup_model_compilation

import wandb    

logger = logging.getLogger(__name__)

class PretrainTrainer(BaseTrainer):
    """
    Unified pretraining trainer for language modeling
    """

    def __init__(self, config, model_config=None, model=None):
        super().__init__(config, model_config, model)
        self.training_flow = TrainingFlow(self)  # ✅ Use training flow utility
        
    def setup_model(self) -> torch.nn.Module:
        """Setup model for pretraining"""
        if self.model is None:
            from myllm.model import GPT
            logger.info(f"Creating pretraining model: {self.model_config.name}")
            self.model = GPT(self.model_config)
        
        self.model.to(self.device)
        self.setup_tokenizer()
        
        # ✅ Use utility for compilation
        self.model = setup_model_compilation(
            self.model, 
            getattr(self.config, "use_compile", False), 
            self.config
        )
        
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
        """Training loop for pretraining using utilities"""
        if self.train_dataloader is None:
            logger.warning("No training data available")
            return

        self.setup_wandb()
        
        from rich.console import Console
        console = Console()
        console.rule(f"[bold green]Starting pretraining for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # ✅ Use utility for progress bar
            progress = create_progress_bar(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            task = progress.add_task(
                f"Epoch {epoch + 1}/{self.config.num_epochs}",
                total=len(self.train_dataloader),
                train_loss=0.0,
                avg_loss=0.0,
                eval_loss=0.0,
                lr=0.0
            )

            with progress:
                # ✅ Use training flow utility for standardized epoch execution
                epoch_loss, num_steps = self.training_flow.run_epoch(epoch, progress, task)

            # ✅ Use training flow utility for end-of-epoch evaluation
            self.training_flow.handle_end_of_epoch(epoch)

        # ✅ Use utilities for final summary
        summary_table = create_training_summary_table(
            "Pretraining Summary", 
            self.config, 
            self.latest_eval_loss, 
            self.best_checkpoint_path, 
            self.wandb_url
        )
        
        console.print(summary_table)
        print_training_completion("Pretraining completed!")
        
        if wandb.run is not None:
            wandb.finish()

    # ✅ REMOVED: All the duplicate methods below are now handled by utilities
    
    # ❌ REMOVED: _create_progress_bar (now in progress_utils)
    # ❌ REMOVED: _update_progress (now in progress_utils) 
    # ❌ REMOVED: _handle_logging_and_evaluation (now in training_flow)
    # ❌ REMOVED: _run_evaluation (now in training_flow)
    # ❌ REMOVED: _handle_end_of_epoch (now in training_flow)
    # ❌ REMOVED: _print_training_summary (now in summary_utils)