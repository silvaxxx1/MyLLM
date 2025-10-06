# examples/train_sft_instruction.py (REFACTORED)
"""
Example for training SFTTrainer with instruction-response data using the new unified architecture.
"""

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, Dataset

    from myllm.Train.configs.SFTConfig import SFTTrainerConfig
    from myllm.Configs.ModelConfig import ModelConfig
    from myllm.Train.factory import create_trainer  # ✅ Use factory
    from myllm.Train.datasets.toy_dataset import get_toy_dataloader  # ✅ Use toy datasets

    print("🚀 Training SFT with Unified Architecture...")

    # -------------------------
    # Trainer & model config
    # -------------------------
    trainer_config = SFTTrainerConfig(
        model_config_name="gpt2-small",
        tokenizer_name="gpt2",
        output_dir="./output_sft_unified",
        num_epochs=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_steps=10,
        max_grad_norm=1.0,
        eval_steps=5,
        save_steps=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_compile=False,  # Disable for testing
        instruction_template="### Instruction:\n{instruction}\n\n### Response:\n{response}",
        # ✅ WandB settings
        wandb_project="sft-instruction-unified",
        wandb_run_name="gpt2-small-sft-unified",
        wandb_tags=["sft", "instruction", "unified"],
        report_to=["wandb"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    model_config = ModelConfig.from_name("gpt2-small")

    # -------------------------
    # Create trainer using factory
    # -------------------------
    trainer = create_trainer("sft", trainer_config, model_config)  # ✅ Use factory
    
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
        "sft", 
        batch_size=trainer_config.batch_size,
        tokenizer=trainer.tokenizer,
        num_samples=50,
        max_length=64
    )
    
    eval_loader = get_toy_dataloader(
        "sft",
        batch_size=trainer_config.batch_size,
        tokenizer=trainer.tokenizer,
        num_samples=10,
        max_length=64
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
    # Start training
    # -------------------------
    print("🎯 Starting SFT training...")
    trainer.train()

    print("✅ SFT training completed!")
    print(f"📁 Output directory: {trainer_config.output_dir}")
    print(f"🏆 Best checkpoint: {trainer.best_checkpoint_path}")