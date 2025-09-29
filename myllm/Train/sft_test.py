# examples/train_sft_instruction.py
"""
Example for training SFTInstructionTrainer with instruction-response data.
Handles:
 - device setup before model
 - proper dataloader & collate_fn
 - response-masked labels
 - optional torch.compile
 - W&B logging
"""

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, Dataset

    from myllm.Train.configs.SFTConfig import SFTTrainerConfig
    from myllm.Configs.ModelConfig import ModelConfig
    from myllm.Train.sft_trainer import SFTInstructionTrainer

    # -------------------------
    # Collate function (robust for SFT)
    # -------------------------
    def collate_fn(batch, tokenizer, instruction_template="### Instruction:\n{instruction}\n\n### Response:\n{response}"):
        """
        batch: list of dicts with {"instruction": str, "response": str}
        tokenizer: TokenizerWrapper with batch_encode or encode
        Returns dict with input_ids, attention_mask, labels (masked for instruction)
        """
        texts = []
        for item in batch:
            instr = item.get("instruction", "")
            resp = item.get("response", "")
            formatted = instruction_template.format(instruction=instr, response=resp)
            texts.append(formatted)

        # Use batch_encode if available
        if hasattr(tokenizer, "batch_encode"):
            encoded = tokenizer.batch_encode(texts, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        else:
            # fallback
            encs = [tokenizer.encode(t, return_tensors="pt").squeeze(0) for t in texts]
            max_len = max([e.size(0) for e in encs])
            padded, masks = [], []
            for e in encs:
                pad_len = max_len - e.size(0)
                padded.append(torch.cat([e, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)], dim=0))
                masks.append(torch.cat([torch.ones(e.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)], dim=0))
            input_ids = torch.stack(padded, dim=0)
            attention_mask = torch.stack(masks, dim=0)

        # Create labels using response mask
        labels = torch.full_like(input_ids, -100)
        for i in range(input_ids.size(0)):
            text = texts[i]
            response_start = text.find("### Response:")
            if response_start != -1:
                prefix = text[:response_start + len("### Response:")]
                prefix_ids = tokenizer.encode(prefix)
                labels[i, len(prefix_ids):] = input_ids[i, len(prefix_ids):]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # -------------------------
    # Dummy instruction-response dataset
    # -------------------------
    class InstructionDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    # -------------------------
    # Example data
    # -------------------------
    sample_data = [
        {"instruction": "Translate to French", "response": "Bonjour le monde"},
        {"instruction": "Summarize the text", "response": "Short summary."},
        {"instruction": "Answer the question", "response": "42"},
        {"instruction": "Write a poem", "response": "Roses are red..."},
    ]

    # -------------------------
    # Trainer & model config
    # -------------------------
    trainer_config = SFTTrainerConfig(
        model_config_name="gpt2-small",
        tokenizer_name="gpt2",
        output_dir="./output_sft",
        num_epochs=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        wandb_project="sft-instruction",
        wandb_run_name="gpt2-small-sft",
        wandb_tags=["sft", "instruction"],
        report_to=["wandb"],
        eval_steps=5,
        save_steps=10,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        max_grad_norm=1.0,
        instruction_template="### Instruction:\n{instruction}\n\n### Response:\n{response}"
    )

    model_config = ModelConfig.from_name("gpt2-small")

    # -------------------------
    # Init trainer
    # -------------------------
    trainer = SFTInstructionTrainer(trainer_config, model_config=model_config)

    # Set device BEFORE setup_model
    trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.global_step = 0

    # Setup model & tokenizer
    trainer.setup_model()
    trainer.model.to(trainer.device)

    # Optional compile
    try:
        if getattr(torch, "compile", None) is not None and torch.cuda.is_available():
            print("Compiling model...")
            trainer.model = torch.compile(trainer.model)
    except Exception as e:
        print(f"torch.compile failed: {e}")

    # -------------------------
    # Dataloader
    # -------------------------
    dataset = InstructionDataset(sample_data)
    train_loader = DataLoader(
        dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer=trainer.tokenizer, instruction_template=trainer_config.instruction_template)
    )

    trainer.train_dataloader = train_loader
    trainer.eval_dataloader = train_loader  # for example purposes

    # -------------------------
    # Optimizer & scheduler
    # -------------------------
    trainer.setup_optimizer()

    # -------------------------
    # Start training
    # -------------------------
    print("Starting SFT training...")
    trainer.train()

    print("Training complete!")
    print(f"Best checkpoint: {trainer.best_checkpoint_path if trainer.best_checkpoint_path else 'N/A'}")
