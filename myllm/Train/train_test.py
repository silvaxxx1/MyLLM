# examples/train_with_existing_project_fixed.py
"""
Fixed example for training using the MyLLM Trainer.
Key fixes:
 - set device in TrainerConfig before Trainer init
 - create dataloaders before setup_optimizer so scheduler can be created
 - proper next-token labels (shift + pad -> -100)
 - safe tokenizer fallback if batch APIs differ
 - pass attention_mask to model forward if present
 - initialize global_step
"""

if __name__ == "__main__":
    import math
    import torch
    from torch.utils.data import DataLoader, Dataset

    from myllm.Configs.ModelConfig import ModelConfig
    from myllm.Train.configs.TempConfig import TrainerConfig
    from myllm.Train.trainer import Trainer

    # -------------------------
    # Collate function (robust)
    # -------------------------
    def collate_fn(batch, tokenizer):
        texts = []
        for item in batch:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            else:
                texts.append(item)

        if hasattr(tokenizer, "batch_encode"):
            encoded = tokenizer.batch_encode(texts, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask", None)
        else:
            encs = [tokenizer.encode(t, return_tensors="pt").squeeze(0) for t in texts]
            max_len = max([e.size(0) for e in encs])
            padded = []
            masks = []
            for e in encs:
                pad_len = max_len - e.size(0)
                if pad_len > 0:
                    pad_val = tokenizer.pad_token or tokenizer.eos_token_id
                    padded_e = torch.cat([e, torch.full((pad_len,), pad_val, dtype=torch.long)], dim=0)
                    mask = torch.cat([torch.ones(e.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
                else:
                    padded_e = e
                    mask = torch.ones(e.size(0), dtype=torch.long)
                padded.append(padded_e)
                masks.append(mask)
            input_ids = torch.stack(padded, dim=0)
            attention_mask = torch.stack(masks, dim=0)

        # Build next-token labels (shift left)
        labels = input_ids.clone()
        if labels.size(1) > 1:
            labels[:, :-1] = input_ids[:, 1:].clone()
            labels[:, -1] = -100
        else:
            labels[:] = -100

        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # -------------------------
    # Simple text-only dataset
    # -------------------------
    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return {"text": self.texts[idx]}

    # -------------------------
    # Trainer configuration
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
        max_grad_norm=1.0,
    )

    # ✅ Set device correctly before Trainer initialization
    trainer_config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Model config
    # -------------------------
    model_config = ModelConfig.from_name("gpt2-small")
    model_config.learning_rate = 3e-4
    model_config.dropout = 0.1

    # -------------------------
    # Initialize trainer
    # -------------------------
    trainer = Trainer(trainer_config, model_config=model_config)
    trainer.global_step = 0  # ensure exists before training

    # Setup model and tokenizer
    trainer.setup_model()
    trainer.model.to(trainer.device)

    # Optional: compile model
    try:
        if getattr(torch, "compile", None) is not None and torch.cuda.is_available():
            print("Compiling model with torch.compile() ...")
            trainer.model = torch.compile(trainer.model, backend="inductor", mode="default")
    except Exception as e:
        print(f"torch.compile() failed (continuing without compile): {e}")

    # -------------------------
    # Sample texts
    # -------------------------
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

    # -------------------------
    # Dataset & DataLoader
    # -------------------------
    dataset = TextDataset(texts)
    train_loader = DataLoader(
        dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer=trainer.tokenizer)
    )

    trainer.train_dataloader = train_loader
    trainer.eval_dataloader = train_loader

    # Setup optimizer/scheduler
    trainer.setup_optimizer()

    # -------------------------
    # Run training
    # -------------------------
    print("Starting training...")
    trainer.train()

    # -------------------------
    # Final output
    # -------------------------
    print("\nTraining completed!")
    print(f"Output directory: {trainer_config.output_dir}")
    print(f"Best checkpoint saved at: {trainer.best_checkpoint_path if getattr(trainer, 'best_checkpoint_path', None) else 'N/A'}")
