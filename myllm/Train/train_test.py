# examples/train_with_existing_project_fixed.py (REFACTORED)
"""
Fixed example for training using the MyLLM unified Trainer architecture.
"""

if __name__ == "__main__":
    import torch
    from myllm.Configs.ModelConfig import ModelConfig
    from myllm.Train.configs.TrainerConfig import TrainerConfig
    from myllm.Train.factory import create_trainer  # ✅ Use factory
    from myllm.Train.datasets.toy_dataset import get_toy_dataloader  # ✅ Use toy datasets

    print("🚀 Training Pretraining with Unified Architecture...")

    # -------------------------
    # Trainer configuration
    # -------------------------
    trainer_config = TrainerConfig(
        model_config_name="gpt2-small",
        tokenizer_name="gpt2",
        output_dir="./output_pretrain_unified",
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_steps=10,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        device="cuda" if torch.cuda.is_available() else "cpu",  # ✅ Set device in config
        use_compile=False,  # Disable for testing
        # ✅ WandB settings
        wandb_project="my-gpt-training-unified",
        wandb_run_name="gpt2-small-pretrain-unified",
        wandb_tags=["pretraining", "gpt2", "unified"],
        report_to=["wandb"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    # -------------------------
    # Model config
    # -------------------------
    model_config = ModelConfig.from_name("gpt2-small")
    model_config.learning_rate = 3e-4
    model_config.dropout = 0.1

    # -------------------------
    # Initialize trainer using factory
    # -------------------------
    trainer = create_trainer("pretrain", trainer_config, model_config)  # ✅ Use factory
    
    # Setup model and tokenizer (handles device internally)
    trainer.setup_model()
    print("✅ Model setup complete")

    # Test tokenizer
    test_text = "Hello world"
    encoded = trainer.tokenizer.encode(test_text)
    print(f"✅ Tokenizer test: '{test_text}' -> {encoded}")

    # -------------------------
    # Create dataloaders using toy datasets
    # -------------------------
    train_loader = get_toy_dataloader(
        "pretrain", 
        batch_size=trainer_config.batch_size,
        tokenizer=trainer.tokenizer,
        num_samples=100,
        max_length=128
    )
    
    eval_loader = get_toy_dataloader(
        "pretrain",
        batch_size=trainer_config.batch_size,
        tokenizer=trainer.tokenizer,
        num_samples=20,
        max_length=128
    )

    # Test batch
    test_batch = next(iter(train_loader))
    print(f"✅ Batch test: input_ids shape {test_batch['input_ids'].shape}")

    # Setup data
    trainer.setup_data(train_loader, eval_loader)
    print("✅ Data setup complete")

    # Setup optimizer
    trainer.setup_optimizer()
    print("✅ Optimizer setup complete")

    # -------------------------
    # Run training
    # -------------------------
    print("🎯 Starting pretraining...")
    trainer.train()

    # -------------------------
    # Final output
    # -------------------------
    print("\n✅ Training completed!")
    print(f"📁 Output directory: {trainer_config.output_dir}")
    print(f"🏆 Best checkpoint: {trainer.best_checkpoint_path}")