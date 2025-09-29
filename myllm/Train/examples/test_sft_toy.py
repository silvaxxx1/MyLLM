# examples/test_sft_toy.py (FIXED)
"""
Complete test script for SFT with toy dataset
"""

if __name__ == "__main__":
    import torch
    from myllm.Configs.ModelConfig import ModelConfig
    from myllm.Train.configs.SFTConfig import SFTTrainerConfig
    from myllm.Train.factory import create_trainer
    from myllm.Train.datasets.toy_dataset import get_toy_dataloader  # ✅ FIXED: plural 'datasets'

    print("🚀 Testing SFT with Toy Dataset...")

    # Configuration with WandB settings
    trainer_config = SFTTrainerConfig(
        model_config_name="gpt2-small",
        tokenizer_name="gpt2",
        output_dir="./output_sft_toy",
        num_epochs=2,
        batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_steps=10,
        max_grad_norm=1.0,
        eval_steps=10,
        save_steps=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_compile=False,
        instruction_template="### Instruction:\n{instruction}\n\n### Response:\n{response}",
        # ✅ WandB specific settings
        wandb_project="myllm-sft-test",
        wandb_run_name="sft-toy-experiment", 
        wandb_tags=["sft", "toy-dataset", "test"],
        report_to=["wandb"],
    )

    model_config = ModelConfig.from_name("gpt2-small")

    # Create trainer
    trainer = create_trainer("sft", trainer_config, model_config)
    
    # Setup model and tokenizer
    trainer.setup_model()
    print("✅ Model setup complete")

    # Test tokenizer
    test_text = "Hello world"
    encoded = trainer.tokenizer.encode(test_text)
    print(f"✅ Tokenizer test: '{test_text}' -> {encoded}")

    # Create toy dataloaders
    train_loader = get_toy_dataloader(
        "sft", 
        batch_size=trainer_config.batch_size,
        tokenizer=trainer.tokenizer,
        num_samples=50
    )
    
    eval_loader = get_toy_dataloader(
        "sft",
        batch_size=trainer_config.batch_size,
        tokenizer=trainer.tokenizer,
        num_samples=10
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
    print("🎯 Starting SFT training...")
    trainer.train()

    print("✅ SFT training test completed!")
    print(f"📁 Output directory: {trainer_config.output_dir}")
    print(f"🏆 Best checkpoint: {trainer.best_checkpoint_path}")