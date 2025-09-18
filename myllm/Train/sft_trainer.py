from .base_trainer import BaseTrainer 
from typing import Dict, Any, Optional 
import torch
import torch.nn as nn 

# trainer/sft_trainer.py
class SFTTrainer(BaseTrainer):
    """Supervised Fine-Tuning Trainer"""
    
    def __init__(self, config, model: Optional[nn.Module] = None):
        super().__init__(config, model)
        # SFT-specific initialization
    
    def setup_model(self) -> nn.Module:
        # TODO: Implement SFT model setup with instruction formatting
        pass
    
    def setup_data(self):
        # TODO: Implement SFT data setup with instruction/response formatting
        pass
    
    def setup_optimizer(self):
        # TODO: Implement SFT optimizer setup (similar to base with PEFT support)
        pass
    
    def train_step(self, batch) -> Dict[str, Any]:
        # TODO: Implement SFT training step with instruction masking
        pass
    
    def evaluate(self) -> Dict[str, float]:
        # TODO: Implement SFT evaluation with generation metrics
        pass
    
    def _train_loop(self):
        # TODO: Implement SFT training loop
        pass 

##############################################################################
import torch.nn as nn
import logging 


logger = logging.getLogger(__name__)

# trainer/sft_trainer.py
class SFTTrainer(BaseTrainer):
    """Supervised Fine-Tuning trainer for instruction following"""
    
    def __init__(self, config, model_config=None, model=None):
        super().__init__(config, model_config, model)
    
    def setup_model(self) -> nn.Module:
        """Setup model for SFT (same as base but with instruction formatting)"""
        if self.model is None:
            from model import GPT
            from Tokenizers.factory import get_tokenizer
            from Tokenizers.wrapper import TokenizerWrapper
            
            logger.info(f"Creating SFT model: {self.model_config.name}")
            self.model = GPT(self.model_config)
            
            tokenizer_raw = get_tokenizer(self.config.tokenizer_name)
            self.tokenizer = TokenizerWrapper(tokenizer_raw)
            
            if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token_id", 0)
        
        return self.model
    
    def setup_data(self):
        """Setup SFT datasets with instruction formatting"""
        logger.info("Setting up SFT datasets...")
        # TODO: Implement SFT data loading with instruction/response formatting
        self.train_dataloader = None
        self.eval_dataloader = None
        logger.info("SFT datasets setup complete")
    
    def setup_optimizer(self):
        """Setup optimizer for SFT (same as base)"""
        if self.model is None:
            raise ValueError("Model must be setup before optimizer")
        
        if self.config.optimizer_type.value == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.model_config.learning_rate,
                weight_decay=self.model_config.weight_decay,
                betas=(self.model_config.beta1, self.model_config.beta2)
            )
        
        logger.info("SFT optimizer setup complete")
    
    def train_step(self, batch) -> Dict[str, Any]:
        """SFT training step with instruction masking"""
        self.model.train()
        
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        logits = self.model(batch["input_ids"])
        
        # Use labels for SFT (instructions are typically masked)
        labels = batch.get("labels", batch["input_ids"][:, 1:].contiguous())
        if labels.dim() == 2:  # If labels don't include the shift
            logits = logits[:, :-1].contiguous()
        
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            labels.view(-1),
            ignore_index=-100  # Ignore instruction tokens
        )
        
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        
        loss.backward()
        
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        return {"loss": loss.item()}
    
    def evaluate(self) -> Dict[str, float]:
        """SFT evaluation"""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                logits = self.model(batch["input_ids"])
                labels = batch.get("labels", batch["input_ids"][:, 1:].contiguous())
                
                if labels.dim() == 2:
                    logits = logits[:, :-1].contiguous()
                
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1),
                    ignore_index=-100
                )
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {"eval_loss": avg_loss}
    
    def _train_loop(self):
        """SFT training loop (same as base)"""
        if self.train_dataloader is None:
            logger.warning("No SFT training data available")
            return
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0
            num_steps = 0
            
            self.log_metrics({
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
            }, self.global_step)
            
            for step, batch in enumerate(self.train_dataloader):
                step_results = self.train_step(batch)
                epoch_loss += step_results["loss"]
                num_steps += 1
                self.global_step += 1
                
                if self.should_log():
                    avg_loss = epoch_loss / num_steps
                    metrics = {
                        "train/loss": step_results["loss"],
                        "train/epoch_avg_loss": avg_loss,
                        "train/epoch": epoch,
                        "train/step": self.global_step
                    }
                    self.log_metrics(metrics)
                
                if self.should_evaluate():
                    eval_results = self.evaluate()
                    if eval_results:
                        eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
                        self.log_metrics(eval_metrics)
                        
                        is_best = self.update_best_metric(eval_results)
                        if is_best:
                            self.save_checkpoint(is_best=True)
                
                if self.should_save_checkpoint():
                    self.save_checkpoint()