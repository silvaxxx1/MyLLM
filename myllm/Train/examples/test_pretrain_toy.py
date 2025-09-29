# examples/test_pretrain_toy.py (FIXED)
"""
Complete test script for pretraining with toy dataset
"""

if __name__ == "__main__":
    import torch
    import wandb
    from myllm.Configs.ModelConfig import ModelConfig
    from myllm.Train.configs.TrainerConfig import TrainerConfig
    from myllm.Train.factory import create_trainer
    from myllm.Train.datasets.toy_dataset import get_toy_dataloader

    print("🚀 Testing Pretraining with Toy Dataset...")

    # Configuration with WandB settings
    trainer_config = TrainerConfig(
        model_config_name="gpt2-small",
        tokenizer_name="gpt2",
        output_dir="./output_pretrain_toy",
        num_epochs=2,
        batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_steps=10,
        max_grad_norm=1.0,
        logging_steps=5,
        eval_steps=10,
        save_steps=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_compile=False,
        # ✅ WandB specific settings
        wandb_project="myllm-pretrain-test",
        wandb_run_name="pretrain-toy-experiment",
        wandb_tags=["pretraining", "toy-dataset", "test"],
        report_to=["wandb"],
    )

    # Model config
    model_config = ModelConfig.from_name("gpt2-small")
    model_config.learning_rate = 5e-5
    model_config.dropout = 0.1

    # ✅ FIXED: Don't manually initialize wandb - let the trainer handle it
    # The trainer will call setup_wandb() which handles initialization

    # Create trainer
    trainer = create_trainer("pretrain", trainer_config, model_config)
    
    # Setup model and tokenizer
    trainer.setup_model()
    print("✅ Model setup complete")

    # Test tokenizer
    test_text = "Hello world"
    encoded = trainer.tokenizer.encode(test_text)
    print(f"✅ Tokenizer test: '{test_text}' -> {encoded}")

    # Create toy dataloaders
    train_loader = get_toy_dataloader(
        "pretrain", 
        batch_size=trainer_config.batch_size,
        tokenizer=trainer.tokenizer,
        num_samples=100
    )
    
    eval_loader = get_toy_dataloader(
        "pretrain",
        batch_size=trainer_config.batch_size,
        tokenizer=trainer.tokenizer,
        num_samples=20
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

    # Start training
    print("🎯 Starting pretraining...")
    trainer.train()

    print("✅ Pretraining test completed!")
    print(f"📁 Output directory: {trainer_config.output_dir}")
    print(f"🏆 Best checkpoint: {trainer.best_checkpoint_path}")