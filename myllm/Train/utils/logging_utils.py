# trainer/utils/logging_utils.py
import logging
import os
import json
import time
from typing import Dict, Any, Optional, List

import torch

# Optional imports for logging backends
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


def is_main_process() -> bool:
    """Check if current process is the main one (DDP/DeepSpeed safe)"""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


class LoggingManager:
    """Logging manager for training framework (WandB + TensorBoard + file)"""
    
    def __init__(self, config, save_metrics_json: bool = True):
        self.config = config
        self.backends = []
        self.step = 0
        self.metrics_history = []
        self.metrics_json_path = os.path.join(config.output_dir, "metrics.json")
        self.save_metrics_json = save_metrics_json and is_main_process()
        
        if is_main_process():
            self._setup_backends()
            self._setup_file_logging()

    # -----------------------------
    # Backend setup
    # -----------------------------
    def _setup_backends(self):
        if "wandb" in self.config.report_to and _has_wandb:
            self._init_wandb()
        if "tensorboard" in self.config.report_to and _has_tensorboard:
            self._init_tensorboard()

    def _init_wandb(self):
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
        try:
            log_dir = self.config.tensorboard_log_dir or os.path.join(self.config.output_dir, "tensorboard_logs")
            os.makedirs(log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir)
            self.backends.append("tensorboard")
            logger.info(f"TensorBoard logging initialized: {log_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")

    def _prepare_wandb_config(self) -> Dict[str, Any]:
        import dataclasses
        if dataclasses.is_dataclass(self.config):
            config_dict = dataclasses.asdict(self.config)
        else:
            config_dict = self.config.__dict__.copy()
        # Remove wandb-specific fields to avoid recursion
        for field in list(config_dict.keys()):
            if field.startswith("wandb_"):
                config_dict.pop(field)
        return config_dict

    # -----------------------------
    # Logging metrics
    # -----------------------------
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if not is_main_process():
            return  # Skip logging on non-main processes

        if step is None:
            step = self.step
            self.step += 1

        metrics_with_meta = {**metrics, "timestamp": time.time(), "step": step}
        self.metrics_history.append(metrics_with_meta)

        # Console
        self._log_to_console(metrics, step)
        # Backends
        if "wandb" in self.backends and wandb.run:
            wandb.log(metrics, step=step)
        if "tensorboard" in self.backends and hasattr(self, "tb_writer"):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
        # JSON
        if self.save_metrics_json:
            self._save_metrics_json(metrics_with_meta)

    def _log_to_console(self, metrics: Dict[str, Any], step: int):
        metric_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        logger.info(f"Step {step}: {', '.join(metric_strs)}")

    def _save_metrics_json(self, metrics: Dict[str, Any]):
        try:
            os.makedirs(os.path.dirname(self.metrics_json_path), exist_ok=True)
            with open(self.metrics_json_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save metrics to JSON: {e}")

    # -----------------------------
    # File logging
    # -----------------------------
    def _setup_file_logging(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        log_file = os.path.join(self.config.output_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    # -----------------------------
    # Model artifact logging
    # -----------------------------
    def log_model_artifacts(self, model_path: str, model_name: str = "model"):
        if not is_main_process():
            return
        if "wandb" in self.backends and wandb.run and getattr(self.config, "log_model", True):
            try:
                artifact = wandb.Artifact(name=f"{model_name}-{wandb.run.id}", type="model")
                artifact.add_dir(model_path)
                wandb.log_artifact(artifact)
                logger.info(f"Model artifacts logged: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to log model artifacts: {e}")

    # -----------------------------
    # Cleanup
    # -----------------------------
    def finish(self, final_metrics: Dict[str, Any] = None):
        if not is_main_process():
            return
        if final_metrics:
            self.log_metrics(final_metrics)
        if "wandb" in self.backends and wandb.run:
            wandb.finish()
        if "tensorboard" in self.backends and hasattr(self, "tb_writer"):
            self.tb_writer.close()
        logger.info("Logging finished")
