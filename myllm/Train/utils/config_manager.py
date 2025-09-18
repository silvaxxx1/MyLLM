# trainer/utils/config_manager.py
import yaml
import json
from typing import Any, Type, Union, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Enhanced Configuration manager with validation and DeepSpeed/multi-GPU support"""

    @staticmethod
    def load_config(config_path: Union[str, Path], config_class: Type) -> Any:
        """Load and validate trainer configuration"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load YAML or JSON
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                raw_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        # Instantiate dataclass
        try:
            config = config_class(**raw_config)
        except TypeError as e:
            logger.error(f"Error creating config from {config_path}: {e}")
            raise

        # Validate required fields
        ConfigManager._validate_config(config)

        # Validate device & multi-GPU / DeepSpeed
        ConfigManager._validate_hardware(config)

        return config

    @staticmethod
    def _validate_config(config: Any):
        """Validate required and optional config fields"""
        required_fields = ["model_config_name", "tokenizer_name", "output_dir"]
        for field in required_fields:
            if not getattr(config, field, None):
                raise ValueError(f"Required config field missing: {field}")

        # Example: If DeepSpeed is used, config path must exist
        if getattr(config, "use_deepspeed", False):
            if not getattr(config, "deepspeed_config_path", None):
                raise ValueError("use_deepspeed=True but deepspeed_config_path is not set")
            ds_path = Path(config.deepspeed_config_path)
            if not ds_path.exists():
                raise FileNotFoundError(f"DeepSpeed config not found: {ds_path}")

    @staticmethod
    def _validate_hardware(config: Any):
        """Validate device, multi-GPU, and mixed-precision compatibility"""
        import torch

        # Device auto-detection
        if config.device == "auto":
            config.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mixed precision check
        if config.mixed_precision and config.device != "cuda":
            logger.warning("Mixed precision requested but CUDA not available. Disabling mixed_precision.")
            config.mixed_precision = False

        # Multi-GPU / DDP check
        if getattr(config, "distributed_backend", None):
            if config.distributed_backend.lower() not in ["nccl", "gloo", "mpi"]:
                raise ValueError(f"Unsupported distributed_backend: {config.distributed_backend}")

    @staticmethod
    def save_config(config: Any, config_path: Union[str, Path]):
        """Save configuration to YAML or JSON, converting enums and dataclasses"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        import dataclasses
        if dataclasses.is_dataclass(config):
            config_data = dataclasses.asdict(config)
        else:
            config_data = config.__dict__.copy()

        # Convert enums to strings
        config_data = ConfigManager._convert_enums_to_strings(config_data)

        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        logger.info(f"Config saved to {config_path}")

    @staticmethod
    def _convert_enums_to_strings(data: Any) -> Any:
        """Convert enum values to strings for serialization"""
        if hasattr(data, 'value'):
            return data.value
        elif isinstance(data, dict):
            return {k: ConfigManager._convert_enums_to_strings(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [ConfigManager._convert_enums_to_strings(item) for item in data]
        else:
            return data
