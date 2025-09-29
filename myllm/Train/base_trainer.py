# trainer/base_trainer.py (FIXED - remove duplicate)
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import logging
import os
from torch.amp import autocast, GradScaler
import wandb

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """
    Unified base trainer with AMP, gradient accumulation, and multi-GPU readiness
    """
    
    def __init__(self, config, model_config=None, model=None):
        self.config = config
        self.model_config = model_config
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None
        self.best_model_path = None
        
        # Data
        self.train_dataloader = None
        self.eval_dataloader = None
        
        # Metrics
        self.latest_eval_loss = None
        self.best_checkpoint_path = None
        self.wandb_url = None
        
        # AMP
        self.scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_logging()
        self._setup_device()
        self._setup_seed()
        self._setup_model_config()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _setup_device(self):
        """Setup training device with multi-GPU readiness"""
        device_str = getattr(self.config.device, 'value', str(self.config.device))
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device_str)
        logger.info(f"Using device: {self.device}")
        
        # Multi-GPU setup placeholder
        self.local_rank = getattr(self.config, 'local_rank', 0)
        self.world_size = getattr(self.config, 'world_size', 1)
        self.distributed = self.world_size > 1
    
    def _setup_seed(self):
        """Setup random seed for reproducibility"""
        seed = getattr(self.config, 'seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _setup_model_config(self):
        """Setup model configuration"""
        if self.model_config is None and hasattr(self.config, 'model_config_name'):
            from myllm.Configs.ModelConfig import ModelConfig
            
            if getattr(self.config, 'model_config_path', None):
                self.model_config = ModelConfig.load(self.config.model_config_path)
            else:
                self.model_config = ModelConfig.from_name(self.config.model_config_name)
            
            # Override with trainer config values
            config_overrides = ['learning_rate', 'weight_decay', 'beta1', 'beta2']
            for attr in config_overrides:
                if hasattr(self.config, attr) and getattr(self.config, attr) is not None:
                    setattr(self.model_config, attr, getattr(self.config, attr))
    
    def setup_model(self) -> nn.Module:
        """Default model setup that can be overridden"""
        if self.model is not None:
            self.model.to(self.device)
            return self.model
        
        # Basic model creation - subclasses should override this
        from myllm.model import GPT
        self.model = GPT(self.model_config)
        self.model.to(self.device)
        
        # Optional compilation
        if getattr(self.config, "use_compile", False):
            try:
                logger.info("Compiling model with torch.compile()")
                self.model = torch.compile(self.model)
            except Exception as e:
                logger.error(f"torch.compile() failed: {e}")
        
        return self.model
    
    def setup_tokenizer(self):
        """Setup tokenizer - common for both trainers - KEEP ONLY THIS ONE"""
        from myllm.Tokenizers.factory import get_tokenizer
        from myllm.Tokenizers.wrapper import TokenizerWrapper
        
        tokenizer_name = getattr(self.config, "tokenizer_name", "gpt2")
        tokenizer_raw = get_tokenizer(tokenizer_name)
        self.tokenizer = TokenizerWrapper(tokenizer_raw)
        
        # Handle pad token
        if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)
            self.tokenizer.pad_token = pad_id
            self.tokenizer.pad_token_id = pad_id  # Also set pad_token_id
            logger.warning(f"Pad token not found, defaulting to eos_token_id={pad_id}")
        
        logger.info(f"Tokenizer setup: {self.tokenizer}")
    
    def setup_optimizer(self):
        """Common optimizer setup"""
        if self.model is None:
            raise ValueError("Model must be setup before optimizer")
        
        # AdamW is standard for both
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
            betas=(self.model_config.beta1, self.model_config.beta2)
        )
        
        # Scheduler setup
        if (hasattr(self.config, 'scheduler_type') and 
            self.config.scheduler_type.value == "linear" and 
            hasattr(self.config, "warmup_steps")):
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        else:
            self.scheduler = None
        
        logger.info("Optimizer and scheduler setup complete")
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Common loss computation"""
        return nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
    
    def align_sequence_lengths(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple:
        """Align sequence lengths for loss computation"""
        if logits.size(1) != labels.size(1):
            min_len = min(logits.size(1), labels.size(1))
            logits = logits[:, :min_len].contiguous()
            labels = labels[:, :min_len].contiguous()
        return logits, labels
    
    # REMOVED DUPLICATE setup_tokenizer METHOD HERE
    
    def training_step(self, batch) -> Dict[str, Any]:
        """Standard training step with AMP and gradient accumulation"""
        self.model.train()
        batch = self._prepare_batch(batch)
        
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            logits = self.model(batch["input_ids"])
            labels = self._get_labels(batch)
            logits, labels = self.align_sequence_lengths(logits, labels)
            
            loss = self.compute_loss(logits, labels)
            loss = loss / max(1, self.config.gradient_accumulation_steps)
        
        self.scaler.scale(loss).backward()
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
        
        return {"loss": float(loss.item())}
    
    def setup_wandb(self):
        """Setup WandB logging if enabled"""
        if (hasattr(self.config, 'report_to') and 
            "wandb" in self.config.report_to and 
            wandb is not None):
            
            try:
                # Check if wandb is already initialized
                if wandb.run is None:
                    wandb_config = {
                        "model_config_name": self.config.model_config_name,
                        "tokenizer_name": self.config.tokenizer_name,
                        "num_epochs": self.config.num_epochs,
                        "batch_size": self.config.batch_size,
                        "learning_rate": self.model_config.learning_rate,
                        "max_grad_norm": self.config.max_grad_norm,
                        "warmup_steps": self.config.warmup_steps,
                        "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                    }
                    
                    wandb.init(
                        project=getattr(self.config, 'wandb_project', 'myllm-training'),
                        name=getattr(self.config, 'wandb_run_name', None),
                        notes=getattr(self.config, 'wandb_notes', None),
                        tags=getattr(self.config, 'wandb_tags', None),
                        config=wandb_config
                    )
                
                # Store the URL for later use
                self.wandb_url = wandb.run.url if wandb.run else "N/A"
                logger.info(f"WandB initialized: {self.wandb_url}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.wandb_url = "N/A"
        else:
            self.wandb_url = "N/A"
    
    def evaluation_step(self, batch) -> Dict[str, float]:
        """Standard evaluation step"""
        self.model.eval()
        batch = self._prepare_batch(batch)
        
        with torch.inference_mode(), autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            logits = self.model(batch["input_ids"])
            labels = self._get_labels(batch)
            logits, labels = self.align_sequence_lengths(logits, labels)
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )
            
            return {
                "loss": loss.item(),
                "tokens": (labels != -100).sum().item()
            }
    
    def evaluate(self) -> Dict[str, float]:
        """Standard evaluation loop"""
        if self.eval_dataloader is None:
            return {}
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in self.eval_dataloader:
            step_results = self.evaluation_step(batch)
            total_loss += step_results["loss"]
            total_tokens += step_results["tokens"]
        
        avg_loss = total_loss / max(1, total_tokens)
        self.latest_eval_loss = avg_loss
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {"eval_loss": avg_loss, "perplexity": perplexity}
    
    # Abstract methods for subclass implementation
    @abstractmethod
    def _prepare_batch(self, batch):
        """Prepare batch for training/evaluation"""
        pass
    
    @abstractmethod
    def _get_labels(self, batch):
        """Get labels from batch (different for pretraining vs SFT)"""
        pass
    
    @abstractmethod
    def setup_data(self, train_dataloader=None, eval_dataloader=None):
        """Setup training and validation data"""
        pass
    
    # Training control methods
    def should_log(self) -> bool:
        return self.global_step % self.config.logging_steps == 0
    
    def should_evaluate(self) -> bool:
        return (hasattr(self.config, 'eval_steps') and 
                self.config.eval_steps > 0 and 
                self.eval_dataloader is not None)
    
    def should_save_checkpoint(self) -> bool:
        return hasattr(self.config, 'save_steps') and self.config.save_steps > 0
    
    def update_best_metric(self, metrics: dict, key: str = "eval_loss") -> bool:
        """Update best metric tracking"""
        current_value = metrics.get(key)
        if current_value is None:
            return False
        
        greater_is_better = getattr(self.config, "greater_is_better", False)
        is_best = False
        
        if self.best_metric is None:
            self.best_metric = current_value
            is_best = True
        else:
            if greater_is_better:
                is_best = current_value > self.best_metric
            else:
                is_best = current_value < self.best_metric
            
            if is_best:
                self.best_metric = current_value
        
        if is_best:
            logger.info(f"New best {key}: {self.best_metric:.4f}")
        
        return is_best
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to WandB and console"""
        if step is None:
            step = self.global_step
        
        # WandB logging
        if wandb.run is not None:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"WandB logging failed: {e}")
        
        # Console logging for key metrics
        if "eval_loss" in metrics:
            logger.info(f"Step {step}: Eval Loss: {metrics['eval_loss']:.4f}")
        elif "train/loss" in metrics:
            logger.info(f"Step {step}: Train Loss: {metrics['train/loss']:.4f}")
    
    def save_checkpoint(self, checkpoint_dir: Optional[str] = None, is_best: bool = False):
        """Save checkpoint with training state"""
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                self.config.output_dir, 
                f"checkpoint-{self.global_step}"
            )
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        if self.model:
            model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            torch.save(self.model.state_dict(), model_path)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }
        
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.bin"))
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        if is_best:
            self.best_checkpoint_path = checkpoint_dir
        
        return checkpoint_dir
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load checkpoint and restore training state"""
        # Load model
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(model_path) and self.model:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load training state
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
            if self.scaler and "scaler" in training_state:
                self.scaler.load_state_dict(training_state["scaler"])
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")