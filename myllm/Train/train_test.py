# examples/train_with_existing_project.py
"""
Example of how to train a GPT-style model using the MyLLM framework
with proper dataset handling, batching, and variable-length sequences.
Includes torch.compile() for performance optimization.
"""

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, Dataset

    from myllm.Configs.ModelConfig import ModelConfig
    from myllm.Configs.GenConfig import GenerationConfig
    from myllm.Train.configs.TempConfig import TrainerConfig
    from myllm.Train.trainer import Trainer

    # ============================================================
    # Collate Function
    # ============================================================
    def collate_fn(batch, tokenizer):
        """
        Combines multiple dataset items into a batch with proper padding.

        Args:
            batch (list of dict): Each item has {"input_ids", "labels", "text"}
            tokenizer: Tokenizer for handling padding.

        Returns:
            dict: A batch dictionary containing:
                - input_ids
                - attention_mask
                - labels
        """
        texts = [item["text"] for item in batch]

        # Tokenize and pad to the longest sequence in the batch
        encoded = tokenizer.batch_encode(texts, padding=True, return_tensors="pt")

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Labels are copies of the input for language modeling
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # ============================================================
    # Custom Dataset
    # ============================================================
    class TextDataset(Dataset):
        """
        A simple dataset that wraps a list of raw texts.
        """
        def __init__(self, texts, tokenizer):
            self.texts = texts
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            """
            Return:
                dict with:
                - "input_ids": tokenized text tensor
                - "labels": same as input_ids (for LM)
                - "text": original text string
            """
            text = self.texts[idx]
            enc = self.tokenizer.encode(text, return_tensors="pt").squeeze(0)
            return {"input_ids": enc, "labels": enc, "text": text}

    # ============================================================
    # Trainer Configuration
    # ============================================================
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

    # ============================================================
    # Load Model Config
    # ============================================================
    model_config = ModelConfig.from_name("gpt2-small")
    model_config.learning_rate = 3e-4
    model_config.dropout = 0.1

    # ============================================================
    # Initialize Trainer
    # ============================================================
    trainer = Trainer(trainer_config, model_config=model_config)

    # Setup model & tokenizer
    trainer.setup_model()

    # Move model to device
    trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model.to(trainer.device)

    # ============================================================
    # ✅ Apply torch.compile for performance optimization
    # ============================================================
    print("Compiling model with torch.compile() for optimized performance...")
    trainer.model = torch.compile(trainer.model, backend="inductor", mode="default")

    # Setup optimizer (must be after moving model to device and compiling)
    trainer.setup_optimizer()

    # ============================================================
    # Sample Training Texts
    # ============================================================
    texts = [
        "Hello world!",
        "This is a test sentence.",
        "Another one.",
        "Variable length sequences are tricky.",
        "We are testing GPT training in MyLLM framework.",
        "This sentence is intentionally long to simulate longer contexts in training.",
        "Monitoring loss and learning rate behavior becomes visible with more data.",
        "Finally, we add a few more sentences to make the dataset slightly larger."
    ]

    # ============================================================
    # Create Dataset & DataLoader
    # ============================================================
    dataset = TextDataset(texts, tokenizer=trainer.tokenizer)

    train_loader = DataLoader(
        dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer=trainer.tokenizer)
    )

    # Use same data for evaluation in this example
    trainer.train_dataloader = train_loader
    trainer.eval_dataloader = train_loader

    # ============================================================
    # Run Training
    # ============================================================
    print("Starting training...")
    trainer.train()

    # ============================================================
    # Final Output
    # ============================================================
    print("\nTraining completed!")
    print(f"Output directory: {trainer_config.output_dir}")
    print(f"Best model saved at: {trainer.best_model_path}")

 
