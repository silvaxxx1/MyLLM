
# trainer/base_trainer.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import os

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """Abstract base class for all trainers"""
    
    def __init__(self, config, model: Optional[nn.Module] = None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None
        self.best_model_path = None
        
        # Logging
        from utils.logging_utils import LoggingManager
        self.logging_manager = LoggingManager(config)
        
        self._setup_logging()
        self._setup_device()
        self._setup_seed()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _setup_device(self):
        """Setup training device"""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        logger.info(f"Using device: {self.device}")
    
    def _setup_seed(self):
        """Setup random seed for reproducibility"""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    @abstractmethod
    def setup_model(self) -> nn.Module:
        """Setup and return the model"""
        pass
    
    @abstractmethod
    def setup_data(self):
        """Setup training and validation datasets"""
        pass
    
    @abstractmethod
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        pass
    
    @abstractmethod
    def train_step(self, batch) -> Dict[str, Any]:
        """Execute a single training step"""
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model"""
        pass
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics using the logging manager"""
        self.logging_manager.log_metrics(metrics, step)
    
    def log_predictions(self, predictions: List[Dict[str, Any]], step: int):
        """Log prediction samples"""
        self.logging_manager.log_predictions(predictions, step)
    
    def log_learning_rate(self):
        """Log current learning rate"""
        if self.optimizer and hasattr(self.optimizer, 'param_groups'):
            lr = self.optimizer.param_groups[0]['lr']
            self.log_metrics({"learning_rate": lr}, self.global_step)
    
    def save_checkpoint(self, checkpoint_dir: Optional[str] = None, is_best: bool = False):
        """Save model checkpoint"""
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                self.config.output_dir, 
                f"checkpoint-{self.global_step}"
            )
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if self.model:
            model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            torch.save(self.model.state_dict(), model_path)
            
            if hasattr(self, 'tokenizer') and self.tokenizer:
                self.tokenizer.save_pretrained(checkpoint_dir)
        
        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "config": self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }
        
        if self.optimizer:
            training_state["optimizer"] = self.optimizer.state_dict()
        if self.scheduler:
            training_state["scheduler"] = self.scheduler.state_dict()
            
        torch.save(training_state, 
                  os.path.join(checkpoint_dir, "training_state.bin"))
        
        if self.config.log_model:
            model_name = "best_model" if is_best else f"checkpoint_{self.global_step}"
            self.logging_manager.log_model_artifacts(checkpoint_dir, model_name)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        if is_best:
            self.best_model_path = checkpoint_dir
        
        return checkpoint_dir
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint"""
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(model_path) and self.model:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        training_state_path = os.path.join(checkpoint_dir, "training_state.bin")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)
            self.global_step = training_state.get("global_step", 0)
            self.current_epoch = training_state.get("current_epoch", 0)
            self.best_metric = training_state.get("best_metric", None)
            
            if self.optimizer and "optimizer" in training_state:
                self.optimizer.load_state_dict(training_state["optimizer"])
            if self.scheduler and "scheduler" in training_state:
                self.scheduler.load_state_dict(training_state["scheduler"])
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
    
    def should_save_checkpoint(self) -> bool:
        return (self.config.save_steps > 0 and 
                self.global_step % self.config.save_steps == 0)
    
    def should_evaluate(self) -> bool:
        return (self.config.eval_steps > 0 and 
                self.global_step % self.config.eval_steps == 0)
    
    def should_log(self) -> bool:
        return self.global_step % self.config.logging_steps == 0
    
    def update_best_metric(self, current_metrics: Dict[str, float]) -> bool:
        """Update best metric and return True if improved"""
        if not self.config.metric_for_best_model:
            return False
        
        metric_key = self.config.metric_for_best_model
        if metric_key not in current_metrics:
            return False
        
        current_value = current_metrics[metric_key]
        
        if self.best_metric is None:
            self.best_metric = current_value
            return True
        
        if self.config.greater_is_better:
            improved = current_value > self.best_metric
        else:
            improved = current_value < self.best_metric
        
        if improved:
            self.best_metric = current_value
            logger.info(f"New best {metric_key}: {current_value}")
            return True
        
        return False
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        self.logging_manager.log_system_metrics()
        
        try:
            self.model = self.setup_model()
            self.setup_data()
            self.setup_optimizer()
            
            if self.model:
                self.model = self.model.to(self.device)
                
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                self.log_metrics({
                    "model/total_parameters": total_params,
                    "model/trainable_parameters": trainable_params,
                    "model/non_trainable_parameters": total_params - trainable_params
                }, step=0)
            
            logger.info(f"Training for {self.config.num_epochs} epochs")
            
            self._train_loop()
            
            final_metrics = self.evaluate()
            if final_metrics:
                self.log_metrics({"final/" + k: v for k, v in final_metrics.items()})
            
            if self.config.load_best_model_at_end and self.best_model_path:
                logger.info("Loading best model for final evaluation")
                self.load_checkpoint(self.best_model_path)
                final_metrics = self.evaluate()
                if final_metrics:
                    self.log_metrics({"best/" + k: v for k, v in final_metrics.items()})
            
            self.logging_manager.finish(final_metrics)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.logging_manager.finish({"error": str(e)})
            raise
    
    @abstractmethod
    def _train_loop(self):
        """Actual training loop implementation"""
        pass