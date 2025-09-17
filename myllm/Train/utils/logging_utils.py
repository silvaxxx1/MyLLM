# trainer/utils/logging_utils.py
import logging
import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

try:
    import wandb
    _has_wandb = True
except ImportError:
    _has_wandb = False
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
    _has_tensorboard = True
except ImportError:
    _has_tensorboard = False
    SummaryWriter = None

logger = logging.getLogger(__name__)

class LoggingManager:
    """Centralized logging manager supporting multiple backends"""
    
    def __init__(self, config):
        self.config = config
        self.backends = []
        self.step = 0
        self.metrics_history = []
        
        self._setup_backends()
        self._setup_file_logging()
    
    def _setup_backends(self):
        """Initialize logging backends"""
        if "wandb" in self.config.report_to and _has_wandb:
            self._init_wandb()
        
        if "tensorboard" in self.config.report_to and _has_tensorboard:
            self._init_tensorboard()
    
    def _init_wandb(self):
        """Initialize Weights & Biases"""
        try:
            wandb_config = self._prepare_wandb_config()
            
            wandb.init(
                project=self.config.wandb_project or "ml-training",
                entity=self.config.wandb_entity,
                name=self.config.wandb_run_name,
                tags=self.config.wandb_tags,
                notes=self.config.wandb_notes,
                config=wandb_config,
                resume=self.config.wandb_resume,
                dir=self.config.output_dir
            )
            
            self.backends.append("wandb")
            logger.info("WandB logging initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
    
    def _init_tensorboard(self):
        """Initialize TensorBoard"""
        try:
            log_dir = self.config.tensorboard_log_dir or os.path.join(
                self.config.output_dir, "tensorboard_logs"
            )
            os.makedirs(log_dir, exist_ok=True)
            
            self.tb_writer = SummaryWriter(log_dir)
            self.backends.append("tensorboard")
            logger.info(f"TensorBoard logging initialized: {log_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
    
    def _setup_file_logging(self):
        """Setup file-based logging"""
        log_file = os.path.join(self.config.output_dir, "training.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(file_handler)
        logger.info(f"File logging setup: {log_file}")
    
    def _prepare_wandb_config(self) -> Dict[str, Any]:
        """Prepare config dict for WandB"""
        config_dict = {}
        
        if hasattr(self.config, '__dict__'):
            config_dict = self.config.__dict__.copy()
        else:
            import dataclasses
            if dataclasses.is_dataclass(self.config):
                config_dict = dataclasses.asdict(self.config)
        
        for field in self.config.wandb_config_exclude:
            config_dict.pop(field, None)
        
        wandb_fields = [f for f in config_dict.keys() if f.startswith('wandb_')]
        for field in wandb_fields:
            config_dict.pop(field, None)
        
        return config_dict
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to all backends"""
        if step is None:
            step = self.step
            self.step += 1
        
        metrics_with_meta = {
            **metrics,
            "timestamp": time.time(),
            "step": step
        }
        
        self.metrics_history.append(metrics_with_meta)
        
        if "wandb" in self.backends and wandb.run:
            wandb.log(metrics, step=step)
        
        if "tensorboard" in self.backends and hasattr(self, 'tb_writer'):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
        
        self._log_to_console(metrics, step)
    
    def _log_to_console(self, metrics: Dict[str, Any], step: int):
        """Log key metrics to console"""
        metric_strs = []
        for key, value in metrics.items():
            if isinstance(value, float):
                metric_strs.append(f"{key}={value:.4f}")
            else:
                metric_strs.append(f"{key}={value}")
        
        logger.info(f"Step {step}: {', '.join(metric_strs)}")
    
    def log_model_artifacts(self, model_path: str, model_name: str = "model"):
        """Log model artifacts"""
        if "wandb" in self.backends and wandb.run and self.config.log_model:
            try:
                artifact = wandb.Artifact(
                    name=f"{model_name}-{wandb.run.id}",
                    type="model"
                )
                artifact.add_dir(model_path)
                wandb.log_artifact(artifact)
                logger.info(f"Model artifacts logged: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to log model artifacts: {e}")
    
    def log_predictions(self, predictions: List[Dict[str, Any]], step: int):
        """Log prediction samples"""
        if not self.config.log_predictions:
            return
        
        if "wandb" in self.backends and wandb.run:
            try:
                table = wandb.Table(
                    columns=list(predictions[0].keys()) if predictions else []
                )
                
                for pred in predictions[:100]:
                    table.add_data(*pred.values())
                
                wandb.log({f"predictions/step_{step}": table}, step=step)
                
            except Exception as e:
                logger.warning(f"Failed to log predictions: {e}")
    
    def log_system_metrics(self):
        """Log system metrics"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            metrics = {
                "system/cpu_percent": cpu_percent,
                "system/memory_used_gb": memory.used / (1024**3),
                "system/memory_percent": memory.percent
            }
            
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        metrics[f"system/gpu_{i}_memory_used"] = (
                            torch.cuda.memory_allocated(i) / (1024**3)
                        )
                        metrics[f"system/gpu_{i}_memory_cached"] = (
                            torch.cuda.memory_reserved(i) / (1024**3)
                        )
            except ImportError:
                pass
            
            self.log_metrics(metrics)
            
        except ImportError:
            logger.warning("psutil not available for system monitoring")
        except Exception as e:
            logger.warning(f"Failed to log system metrics: {e}")
    
    def save_metrics_history(self):
        """Save metrics history to file"""
        history_file = os.path.join(self.config.output_dir, "metrics_history.json")
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
            logger.info(f"Metrics history saved: {history_file}")
        except Exception as e:
            logger.warning(f"Failed to save metrics history: {e}")
    
    def generate_report(self, final_metrics: Dict[str, Any] = None):
        """Generate final training report"""
        report_path = os.path.join(self.config.output_dir, "training_report.md")
        
        try:
            with open(report_path, 'w') as f:
                f.write("# Training Report\n\n")
                f.write(f"**Project:** {self.config.wandb_project or 'Unknown'}\n")
                f.write(f"**Model:** {self.config.model_name_or_path}\n")
                f.write(f"**Output Directory:** {self.config.output_dir}\n\n")
                
                f.write("## Configuration\n\n")
                f.write(f"- Learning Rate: {self.config.learning_rate}\n")
                f.write(f"- Batch Size: {self.config.batch_size}\n")
                f.write(f"- Epochs: {self.config.num_epochs}\n")
                f.write(f"- Optimizer: {self.config.optimizer.value}\n\n")
                
                if final_metrics:
                    f.write("## Final Metrics\n\n")
                    for key, value in final_metrics.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                if "wandb" in self.backends and wandb.run:
                    f.write(f"## Links\n\n")
                    f.write(f"- [WandB Run]({wandb.run.url})\n")
                
            logger.info(f"Training report generated: {report_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate report: {e}")
    
    def finish(self, final_metrics: Dict[str, Any] = None):
        """Cleanup and finish logging"""
        self.generate_report(final_metrics)
        self.save_metrics_history()
        
        if "wandb" in self.backends and wandb.run:
            if final_metrics:
                wandb.log(final_metrics)
            wandb.finish()
        
        if "tensorboard" in self.backends and hasattr(self, 'tb_writer'):
            self.tb_writer.close()
        
        logger.info("Logging finished")