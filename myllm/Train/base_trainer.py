from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm

from ..api import LLM  # Your existing LLM class
from ..Configs import ModelConfig  # Your existing ModelConfig
from .configs import TrainingConfig
from .utils.callbacks import CallbackHandler
from .utils.metrics import MetricsTracker
from .utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """
    Abstract base trainer that integrates with your existing LLM and ModelConfig system
    """
    
    def __init__(
        self,
        llm: LLM,
        config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer=None,  # Your TokenizerWrapper
        callbacks: Optional[List] = None,
    ):
        self.llm = llm
        self.model = llm.model  # Access the underlying PyTorch model
        self.model_config = llm.config  # Your existing ModelConfig
        self.config = config
        self.device = llm.device
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        
        # Training state
        self.optimizer: Optional[Optimizer] = None
        self.lr_scheduler: Optional[_LRScheduler] = None
        self.scaler = None  # For mixed precision
        
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        
        # Training utilities
        self.callback_handler = CallbackHandler(callbacks or [])
        self.metrics_tracker = MetricsTracker()
        self.checkpoint_manager = CheckpointManager(
            output_dir=config.output_dir,
            save_total_limit=config.save_total_limit
        )
        
        # Setup mixed precision
        if config.fp16 or config.bf16:
            self.scaler = torch.cuda.amp.GradScaler()
            
    @abstractmethod
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch. Must return dict with at least 'loss' key.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            Dict containing 'loss' and optionally other metrics
        """
        pass
    
    @abstractmethod
    def prepare_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Prepare batch for training (move to device, format, etc.)
        
        Args:
            batch: Raw batch from dataloader
            
        Returns:
            Prepared batch dict
        """
        pass
    
    def setup_training(self):
        """Setup optimizer, scheduler, and other training components"""
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_scheduler()
        
        # Apply any model config overrides
        if self.config.model_config_overrides:
            for key, value in self.config.model_config_overrides.items():
                if hasattr(self.model_config, key):
                    setattr(self.model_config, key, value)
                    logger.info(f"Override model config: {key} = {value}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Training config: {self.config}")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,} total parameters")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters")
        
        self.setup_training()
        
        train_dataloader = self._create_dataloader(self.train_dataset)
        eval_dataloader = self._create_dataloader(self.eval_dataset) if self.eval_dataset else None
        
        self.callback_handler.on_train_begin(self)
        
        try:
            self.model.train()
            
            if self.config.max_epochs > 0:
                self._train_epochs(train_dataloader, eval_dataloader)
            else:
                self._train_steps(train_dataloader, eval_dataloader)
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            self.callback_handler.on_train_end(self)
            
        logger.info("Training completed!")
    
    def _train_epochs(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader]):
        """Training loop by epochs"""
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{self.config.max_epochs}")
            
            self.callback_handler.on_epoch_begin(self)
            
            epoch_metrics = self._run_epoch(train_dataloader, eval_dataloader)
            
            self.callback_handler.on_epoch_end(self, epoch_metrics)
            
            # Check if we should stop early
            if self.global_step >= self.config.max_steps:
                logger.info(f"Reached max_steps ({self.config.max_steps}), stopping training")
                break
    
    def _train_steps(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader]):
        """Training loop by steps"""
        step_iterator = iter(train_dataloader)
        
        with tqdm(total=self.config.max_steps, desc="Training") as pbar:
            while self.global_step < self.config.max_steps:
                try:
                    batch = next(step_iterator)
                except StopIteration:
                    step_iterator = iter(train_dataloader)
                    batch = next(step_iterator)
                
                logs = self._training_step(batch)
                
                pbar.update(1)
                pbar.set_postfix(logs)
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics(logs)
                
                # Evaluation
                if eval_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    self._log_metrics(eval_metrics, prefix="eval")
                
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                self.callback_handler.on_step_end(self, logs)
                self.global_step += 1
    
    def _run_epoch(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader]) -> Dict[str, float]:
        """Run one training epoch"""
        epoch_metrics = {}
        
        with tqdm(train_dataloader, desc=f"Epoch {self.epoch + 1}") as pbar:
            for batch in pbar:
                if self.global_step >= self.config.max_steps:
                    break
                    
                logs = self._training_step(batch)
                
                pbar.set_postfix(logs)
                
                # Update epoch metrics
                for key, value in logs.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics(logs)
                
                # Evaluation
                if eval_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    self._log_metrics(eval_metrics, prefix="eval")
                
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                self.callback_handler.on_step_end(self, logs)
                self.global_step += 1
        
        # Average epoch metrics
        return {key: sum(values) / len(values) for key, values in epoch_metrics.items()}
    
    def _training_step(self, batch: Any) -> Dict[str, float]:
        """Execute one training step"""
        prepared_batch = self.prepare_batch(batch)
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast(enabled=self.config.fp16 or self.config.bf16):
            loss_dict = self.compute_loss(prepared_batch)
            loss = loss_dict['loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (after accumulation)
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            if self.lr_scheduler:
                self.lr_scheduler.step()
        
        # Prepare logs
        logs = {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'step': self.global_step
        }
        
        # Add other metrics from loss_dict
        for key, value in loss_dict.items():
            if key != 'loss':
                logs[key] = value.item() if isinstance(value, torch.Tensor) else value
        
        return logs
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Run evaluation"""
        self.model.eval()
        eval_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                prepared_batch = self.prepare_batch(batch)
                
                with torch.cuda.amp.autocast(enabled=self.config.fp16 or self.config.bf16):
                    loss_dict = self.compute_loss(prepared_batch)
                
                for key, value in loss_dict.items():
                    if key not in eval_metrics:
                        eval_metrics[key] = []
                    eval_metrics[key].append(value.item() if isinstance(value, torch.Tensor) else value)
        
        # Average metrics
        eval_metrics = {f"eval_{key}": sum(values) / len(values) for key, values in eval_metrics.items()}
        
        self.model.train()
        return eval_metrics
    
    def save_checkpoint(self, checkpoint_dir: Optional[str] = None):
        """Save model checkpoint"""
        if checkpoint_dir is None:
            checkpoint_dir = f"{self.config.output_dir}/checkpoint-{self.global_step}"
        
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict (compatible with your LLM.load method)
        model_path = checkpoint_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }
        
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump({k: v for k, v in training_state.items() if v is not None}, f, indent=2, default=str)
        
        # Save configs
        self.config.save(checkpoint_dir / "training_config.json")
        self.model_config.save(checkpoint_dir / "model_config.json")
        
        logger.info(f"Checkpoint saved at {checkpoint_dir}")
        
        self.checkpoint_manager.save_checkpoint(checkpoint_dir)
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load model
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Model loaded from {model_path}")
        
        # Load training state
        training_state_path = checkpoint_dir / "training_state.json"
        if training_state_path.exists():
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
            
            self.global_step = training_state.get('global_step', 0)
            self.epoch = training_state.get('epoch', 0)
            self.best_metric = training_state.get('best_metric', float('inf'))
            
            logger.info(f"Training state loaded from {training_state_path}")
            logger.info(f"Resuming from step {self.global_step}, epoch {self.epoch}")
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer"""
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler"""
        if self.config.lr_scheduler == "constant":
            return None
        
        total_steps = self.config.max_steps if self.config.max_steps > 0 else len(self._create_dataloader(self.train_dataset)) * self.config.max_epochs
        
        if self.config.lr_scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps - self.config.warmup_steps
            )
        elif self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.config.warmup_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.lr_scheduler}")
    
    def _create_dataloader(self, dataset: Dataset) -> DataLoader:
        """Create DataLoader for dataset"""
        if dataset is None:
            return None
            
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            collate_fn=self._get_collate_fn()
        )
    
    def _get_collate_fn(self):
        """Get collate function for DataLoader"""
        # Default collate function - override in subclasses if needed
        return None
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        """Log training metrics"""
        self.metrics_tracker.log_metrics(metrics, self.global_step, prefix)
        
        # Log to console
        log_str = f"Step {self.global_step}"
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_str += f" | {key}: {value:.4f}"
        logger.info(log_str)