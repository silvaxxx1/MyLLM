
# trainer/trainer.py
from base_trainer import BaseTrainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    get_linear_schedule_with_warmup
)
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Trainer(BaseTrainer):
    """Main trainer class for pre-training"""
    
    def __init__(self, config, model: Optional[nn.Module] = None):
        super().__init__(config, model)
        self.train_dataloader = None
        self.eval_dataloader = None
        self.is_distributed = config.distributed
        self.deepspeed_engine = None
        
    def setup_model(self) -> nn.Module:
        """Setup the model for pre-training"""
        if self.model is None:
            logger.info(f"Loading model: {self.config.model_name_or_path}")
            
            model_config = AutoConfig.from_pretrained(self.config.model_name_or_path)
            self.model = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                config=model_config
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.model
    
    def setup_data(self):
        """Setup training and validation datasets"""
        logger.info("Setting up datasets...")
        # Placeholder for data loading - implement based on your data format
        self.train_dataloader = None  # DataLoader(train_dataset, ...)
        self.eval_dataloader = None   # DataLoader(eval_dataset, ...)
        logger.info("Datasets setup complete")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        if self.model is None:
            raise ValueError("Model must be setup before optimizer")
        
        from configs.BaseConfig import OptimizerType
        
        if self.config.optimizer == OptimizerType.ADAMW:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer {self.config.optimizer} not implemented")
        
        if self.train_dataloader is not None:
            total_steps = len(self.train_dataloader) * self.config.num_epochs
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )
        
        logger.info("Optimizer and scheduler setup complete")
    
    def train_step(self, batch) -> Dict[str, Any]:
        """Execute a single training step"""
        self.model.train()
        
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
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
        """Evaluate the model"""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {"eval_loss": avg_loss}
    
    def _train_loop(self):
        """Main training loop implementation"""
        if self.train_dataloader is None:
            logger.warning("No training data available")
            return
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0
            num_steps = 0
            
            # Log epoch start
            self.log_metrics({
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
            }, self.global_step)
            
            for step, batch in enumerate(self.train_dataloader):
                # Training step
                step_results = self.train_step(batch)
                epoch_loss += step_results["loss"]
                num_steps += 1
                self.global_step += 1
                
                # Logging
                if self.should_log():
                    avg_loss = epoch_loss / num_steps
                    metrics = {
                        "train/loss": step_results["loss"],
                        "train/epoch_avg_loss": avg_loss,
                        "train/epoch": epoch,
                        "train/step": self.global_step
                    }
                    
                    # Add learning rate
                    self.log_learning_rate()
                    
                    # Add any additional metrics from train_step
                    for key, value in step_results.items():
                        if key != "loss":
                            metrics[f"train/{key}"] = value
                    
                    self.log_metrics(metrics)
                    
                    # Log system metrics periodically
                    if self.global_step % (self.config.logging_steps * 10) == 0:
                        self.logging_manager.log_system_metrics()
                
                # Evaluation
                if self.should_evaluate():
                    eval_results = self.evaluate()
                    if eval_results:
                        eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
                        self.log_metrics(eval_metrics)
                        
                        # Check if this is the best model
                        is_best = self.update_best_metric(eval_results)
                        
                        # Log prediction samples if configured
                        if self.config.log_predictions and hasattr(self, 'get_prediction_samples'):
                            try:
                                predictions = self.get_prediction_samples(num_samples=10)
                                self.log_predictions(predictions, self.global_step)
                            except Exception as e:
                                logger.warning(f"Failed to log predictions: {e}")
                
                # Save checkpoint
                if self.should_save_checkpoint():
                    checkpoint_path = self.save_checkpoint()
                    
                    # Save best model checkpoint
                    if hasattr(self, '_last_eval_was_best') and self._last_eval_was_best:
                        self.save_checkpoint(is_best=True)
                        self._last_eval_was_best = False
                
                # Update best model flag for next checkpoint
                if 'is_best' in locals():
                    self._last_eval_was_best = is_best
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / num_steps if num_steps > 0 else 0
            self.log_metrics({
                "train/epoch_end_loss": avg_epoch_loss,
                "train/completed_epoch": epoch
            })
            
            # End of epoch evaluation
            if self.eval_dataloader is not None:
                eval_results = self.evaluate()
                if eval_results:
                    eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
                    eval_metrics["eval/epoch"] = epoch
                    self.log_metrics(eval_metrics)
                    
                    # Update best metric
                    is_best = self.update_best_metric(eval_results)
                    if is_best:
                        self.save_checkpoint(is_best=True)
    
    def get_prediction_samples(self, num_samples: int = 10):
        """Get prediction samples for logging - to be overridden by subclasses"""
        return []