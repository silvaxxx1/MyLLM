from .BaseConfig import TrainConfig
from .SFTConfig import SFTConfig 
from .DPOConfig import DPOConfig 
from .PPOConfig import PPOConfig

# Factory function to create appropriate config
def create_config(trainer_type: str, **kwargs) -> TrainConfig:
    """Factory function to create appropriate config based on trainer type."""
    config_map = {
        "sft": SFTConfig,
        "dpo": DPOConfig, 
        "ppo": PPOConfig,
    }
    
    if trainer_type not in config_map:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Choose from {list(config_map.keys())}")
    
    return config_map[trainer_type](**kwargs)


# Example usage:
if __name__ == "__main__":
    # Create SFT config
    sft_config = SFTConfig(
        output_dir="./sft_checkpoints",
        num_train_epochs=3,
        train_batch_size=16,
        learning_rate=2e-5,
        max_seq_length=2048
    )
    
    # Create DPO config  
    dpo_config = DPOConfig(
        output_dir="./dpo_checkpoints",
        num_train_epochs=1,
        train_batch_size=8,
        learning_rate=5e-7,
        beta=0.1
    )
    
    # Create PPO config
    ppo_config = PPOConfig(
        output_dir="./ppo_checkpoints", 
        num_train_epochs=1,
        train_batch_size=8,
        learning_rate=1e-6,
        rollout_batch_size=512,
        reward_model_name="./reward_model"
    )
    
    print("Configs created successfully!")