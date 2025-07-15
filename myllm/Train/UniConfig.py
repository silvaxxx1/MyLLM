
# =====================================
# configs/unified_config.py
"""
Unified Training Configuration

Contains TrainConfig class and factory functions for managing
all training types in a single interface.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import json
from BaseConfig import BaseTrainConfig
from SFTConfig import SFTConfig
from DPOConfig import DPOConfig
from PPOConfig import PPOConfig

@dataclass
class TrainConfig:
    """Unified training configuration that can handle different training types"""
    
    # Training type
    training_type: Literal["sft", "dpo", "ppo"] = "sft"
    
    # Specific configs
    sft_config: Optional[SFTConfig] = None
    dpo_config: Optional[DPOConfig] = None
    ppo_config: Optional[PPOConfig] = None
    
    # Common settings that override specific configs
    model_name_or_path: Optional[str] = None
    output_dir: str = "./outputs"
    run_name: Optional[str] = None
    
    def __post_init__(self):
        # Initialize the appropriate config based on training type
        if self.training_type == "sft" and self.sft_config is None:
            self.sft_config = SFTConfig()
        elif self.training_type == "dpo" and self.dpo_config is None:
            self.dpo_config = DPOConfig()
        elif self.training_type == "ppo" and self.ppo_config is None:
            self.ppo_config = PPOConfig()
        
        # Override specific config settings with common ones
        active_config = self.get_active_config()
        if active_config:
            if self.model_name_or_path:
                # Here you would load the model config
                pass
            active_config.output_dir = self.output_dir
            if self.run_name:
                active_config.wandb_run_name = self.run_name

    def get_active_config(self) -> Optional[BaseTrainConfig]:
        """Get the active configuration based on training type"""
        if self.training_type == "sft":
            return self.sft_config
        elif self.training_type == "dpo":
            return self.dpo_config
        elif self.training_type == "ppo":
            return self.ppo_config
        return None

    def save(self, file_path: str):
        """Save the unified config to file"""
        config_dict = {
            "training_type": self.training_type,
            "model_name_or_path": self.model_name_or_path,
            "output_dir": self.output_dir,
            "run_name": self.run_name
        }
        
        # Add the active config
        active_config = self.get_active_config()
        if active_config:
            config_dict[f"{self.training_type}_config"] = active_config.to_dict()
        
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, file_path: str):
        """Load unified config from file"""
        with open(file_path, "r") as f:
            data = json.load(f)
        
        training_type = data.get("training_type", "sft")
        config = cls(
            training_type=training_type,
            model_name_or_path=data.get("model_name_or_path"),
            output_dir=data.get("output_dir", "./outputs"),
            run_name=data.get("run_name")
        )
        
        # Load the specific config
        if f"{training_type}_config" in data:
            config_data = data[f"{training_type}_config"]
            if training_type == "sft":
                config.sft_config = SFTConfig.from_dict(config_data)
            elif training_type == "dpo":
                config.dpo_config = DPOConfig.from_dict(config_data)
            elif training_type == "ppo":
                config.ppo_config = PPOConfig.from_dict(config_data)
        
        return config

# Factory functions for easy config creation
def create_sft_config(
    model_config=None,
    train_dataset_path: str = "",
    max_epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    **kwargs
) -> SFTConfig:
    """Factory function to create SFT configuration"""
    return SFTConfig(
        model_config=model_config,
        train_dataset_path=train_dataset_path,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        **kwargs
    )

def create_dpo_config(
    model_config=None,
    train_dataset_path: str = "",
    beta: float = 0.1,
    max_epochs: int = 1,
    learning_rate: float = 1e-6,
    batch_size: int = 4,
    **kwargs
) -> DPOConfig:
    """Factory function to create DPO configuration"""
    return DPOConfig(
        model_config=model_config,
        train_dataset_path=train_dataset_path,
        beta=beta,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        **kwargs
    )

def create_ppo_config(
    model_config=None,
    query_dataset_path: str = "",
    reward_model_path: str = "",
    ppo_epochs: int = 4,
    learning_rate: float = 1e-5,
    batch_size: int = 64,
    **kwargs
) -> PPOConfig:
    """Factory function to create PPO configuration"""
    return PPOConfig(
        model_config=model_config,
        query_dataset_path=query_dataset_path,
        reward_model_path=reward_model_path,
        ppo_epochs=ppo_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        **kwargs
    )

# =====================================
# Example usage in your training scripts:

"""
# train_sft.py
from configs import SFTConfig, ModelConfig

model_config = ModelConfig.from_name("llama3-8b")
sft_config = SFTConfig(
    model_config=model_config,
    train_dataset_path="./data/sft_train.json",
    max_epochs=3,
    use_lora=True
)

# train_dpo.py  
from configs import DPOConfig, ModelConfig

model_config = ModelConfig.from_name("llama3-8b")
dpo_config = DPOConfig(
    model_config=model_config,
    train_dataset_path="./data/dpo_train.json",
    beta=0.1,
    learning_rate=1e-6
)

# train_ppo.py
from configs import PPOConfig, ModelConfig

model_config = ModelConfig.from_name("llama3-8b")
ppo_config = PPOConfig(
    model_config=model_config,
    query_dataset_path="./data/ppo_queries.json",
    reward_model_path="./models/reward_model"
)

# main_train.py - unified approach
from configs import TrainConfig, create_sft_config, ModelConfig

model_config = ModelConfig.from_name("llama3-8b")
sft_config = create_sft_config(
    model_config=model_config,
    train_dataset_path="./data/sft_train.json",
    use_lora=True
)

unified_config = TrainConfig(
    training_type="sft",
    model_name_or_path="./models/base_model",
    output_dir="./outputs/sft_run",
    sft_config=sft_config
)
"""