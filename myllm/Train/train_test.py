

# examples/train_example.py
"""
Example usage of the training framework
"""

if __name__ == "__main__":
    import os
    from trainer import Trainer
    from configs import TrainerConfig
    from utils.config_manager import ConfigManager


    
    # Create configuration
    config = TrainerConfig(
        model_name_or_path="gpt2",
        output_dir="./output",
        num_epochs=3,
        batch_size=8,
        learning_rate=5e-5,
        
        # Logging configuration
        wandb_project="ml-training-demo",
        wandb_run_name="gpt2-pretraining-example",
        wandb_tags=["pretraining", "gpt2", "demo"],
        wandb_notes="Demo run with enhanced logging framework",
        
        # Monitoring
        report_to=["wandb", "tensorboard"],
        log_model=True,
        log_predictions=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        
        # Training parameters
        logging_steps=50,
        eval_steps=500,
        save_steps=1000,
        warmup_steps=100,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        
        # System
        seed=42,
        mixed_precision=True,
    )
    
    # Save config for reproducibility
    ConfigManager.save_config(config, "output/config.yaml")
    


