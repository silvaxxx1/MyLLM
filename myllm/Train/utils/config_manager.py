
# trainer/utils/config_manager.py
import yaml
import json
from typing import Dict, Any, Type, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Utility class for managing configuration files"""
    
    @staticmethod
    def load_config(config_path: Union[str, Path], config_class: Type) -> Any:
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        try:
            return config_class(**config_data)
        except TypeError as e:
            logger.error(f"Error creating config from {config_path}: {e}")
            raise
    
    @staticmethod
    def save_config(config: Any, config_path: Union[str, Path]):
        """Save configuration to YAML or JSON file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(config, '__dict__'):
            config_data = config.__dict__.copy()
        else:
            import dataclasses
            if dataclasses.is_dataclass(config):
                config_data = dataclasses.asdict(config)
            else:
                raise ValueError("Config must be a dataclass or have __dict__ attribute")
        
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
        """Recursively convert enum values to strings for serialization"""
        if hasattr(data, 'value'):
            return data.value
        elif isinstance(data, dict):
            return {k: ConfigManager._convert_enums_to_strings(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [ConfigManager._convert_enums_to_strings(item) for item in data]
        else:
            return data
