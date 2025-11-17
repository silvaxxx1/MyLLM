# trainer/sft_trainer.py (REFACTORED)
import logging
from typing import Dict, Any
import torch

from .base_trainer import BaseTrainer
from .utils.progress_utils import create_progress_bar
from .utils.training_flow import TrainingFlow
from .utils.summary_utils import create_training_summary_table, print_training_completion
from .utils.model_utils import setup_model_compilation, load_pretrained_weights

import wandb

logger = logging.getLogger(__name__)

class SFTTrainer(BaseTrainer):
    """
    Unified SFT trainer for instruction following
    """

    def __init__(self, config, model_config=None, model=None):
        super().__init__(config, model_config, model)
        self.training_flow = TrainingFlow(self)  # ✅ Use training flow utility
        
        self.instruction_template = getattr(config, 'instruction_template', 
                                          "### Instruction:\n{instruction}\n\n### Response:\n{response}")
        self.response_marker = getattr(config, 'response_template', "### Response:")

    def setup_model(self) -> torch.nn.Module:
        """Setup model for SFT with optional pretrained weights"""
        from myllm.api import LLM

        if self.model is not None:
            logger.info("Using externally provided model")
            self.model.to(self.device)
            self.setup_tokenizer()
            return self.model

        if self.model_config is None:
            raise ValueError("model_config must be provided to create a model")

        logger.info(f"Creating LLM: {self.model_config.name} on {self.device}")
        self.llm = LLM(config=self.model_config, device=self.device)

        # ✅ Use utility for pretrained weights loading
        load_pretrained_weights(
            self.llm, 
            getattr(self.config, "pretrained_variant", None), 
            "gpt2"
        )

        self.model = self.llm.model
        self.setup_tokenizer()
        self.model.to(self.device)

        # ✅ Use utility for model compilation
        self.model = setup_model_compilation(
            self.model, 
            getattr(self.config, "use_compile", False), 
            self.config
        )

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
        """Create response mask for SFT training - KEEP THIS SFT-SPECIFIC"""
        # start as the same as input_ids
        labels = input_ids.clone()
        
        # loop over each input
        for i in range(input_ids.size(0)):
            # tokenize the full sequence to text
            text = self.tokenizer.decode(input_ids[i], skip_special_tokens=False)
            # find the position of the response marker
            response_start = text.find(self.response_marker)
            # if the response marker is found, mask the tokens before it
            if response_start != -1:
                # get the text before the response marker
                prefix = text[:response_start + len(self.response_marker)]
                # encode the prefix
                prefix_tokens = self.tokenizer.encode(prefix)
                # mask the tokens before the response marker
                mask_until = len(prefix_tokens)
                if mask_until < labels.size(1):
                    labels[i, :mask_until] = -100
        
        # Apply attention mask
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)
            
        return labels

    def train(self):
        """Training loop for SFT using utilities"""
        if self.train_dataloader is None:
            logger.warning("No training data available")
            return

        self.setup_wandb()
        
        from rich.console import Console
        console = Console()
        console.rule(f"[bold green]Starting SFT Training for {self.config.num_epochs} epochs")

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
            "SFT Training Summary", 
            self.config, 
            self.latest_eval_loss, 
            self.best_checkpoint_path, 
            self.wandb_url
        )
        
        console.print(summary_table)
        print_training_completion("SFT Training Completed!")
        
        if wandb.run is not None:
            wandb.finish()

    # ✅ REMOVED: All the duplicate methods below are now handled by utilities
    
    # ❌ REMOVED: _create_progress_bar (now in progress_utils)
    # ❌ REMOVED: _update_progress (now in progress_utils) 
    # ❌ REMOVED: _handle_logging_and_evaluation (now in training_flow)
    # ❌ REMOVED: _run_evaluation (now in training_flow)
    # ❌ REMOVED: _handle_end_of_epoch (now in training_flow)
    # ❌ REMOVED: _print_training_summary (now in summary_utils)