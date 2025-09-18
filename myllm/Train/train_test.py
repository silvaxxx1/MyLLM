# examples/train_with_existing_project.py
"""
Example of how to use the training framework with your existing project
with proper dataset handling and batching for variable-length sequences.
"""

if __name__ == "__main__":
    from torch.utils.data import DataLoader, Dataset
    import torch

    from myllm.Configs.ModelConfig import ModelConfig
    from myllm.Configs.GenConfig import GenerationConfig
    from myllm.Train.configs.TrainerConfig import TrainerConfig
    from myllm.Train.trainer import Trainer

    # -------------------------
    # Collate function
    # -------------------------
    def collate_fn(batch, tokenizer):
        texts = [item["text"] for item in batch]
        encoded = tokenizer.batch_encode(texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # -------------------------
    # Dataset
    # -------------------------
    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer):
            self.texts = texts
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            enc = self.tokenizer.encode(text, return_tensors="pt").squeeze(0)
            return {"input_ids": enc, "labels": enc, "text": text}

    # -------------------------
    # Trainer Config
    # -------------------------
    trainer_config = TrainerConfig(
        model_config_name="gpt2-small",
        tokenizer_name="gpt2",
        output_dir="./output",
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        wandb_project="my-gpt-training",
        wandb_run_name="gpt2-small-experiment",
        wandb_tags=["pretraining", "gpt2"],
        report_to=["wandb"],
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        max_grad_norm=1.0
    )

    # -------------------------
    # Load model config
    # -------------------------
    model_config = ModelConfig.from_name("gpt2-small")
    model_config.learning_rate = 3e-4
    model_config.dropout = 0.1

    # -------------------------
    # Initialize Trainer
    # -------------------------
    trainer = Trainer(trainer_config, model_config=model_config)
    trainer.setup_model()  # initializes GPT and tokenizer

    # Move model to device
    trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model.to(trainer.device)

    # Setup optimizer (must be after model is on device)
    trainer.setup_optimizer()

    # -------------------------
    # Sample training texts
    # -------------------------
    texts = [
        "Hello world!",
        "This is a test sentence.",
        "Another one.",
        "Variable length sequences are tricky.",
        "We are testing GPT training in MyLLM framework.",
        "Padding and attention masks must work correctly."
    ]

    # -------------------------
    # Create dataset & DataLoader
    # -------------------------
    dataset = TextDataset(texts, tokenizer=trainer.tokenizer)
    train_loader = DataLoader(
        dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer=trainer.tokenizer)
    )

    trainer.train_dataloader = train_loader
    trainer.eval_dataloader = train_loader  # simple evaluation set

    # -------------------------
    # Run training
    # -------------------------
    trainer._train_loop()

    print("\nTraining completed!")
    print(f"Output directory: {trainer_config.output_dir}")
    print(f"Best model saved at: {trainer.best_model_path}")
