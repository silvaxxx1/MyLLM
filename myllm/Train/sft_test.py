# examples/train_sft_with_existing_project.py
"""
Example of how to train a GPT-style model for SFT (Supervised Fine-Tuning)
using the MyLLM framework with instruction-response pairs.
Includes torch.compile() for performance optimization.
"""

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, Dataset

    from myllm.Configs.ModelConfig import ModelConfig
    from myllm.Configs.GenConfig import GenerationConfig
    from myllm.Train.configs.TempConfig import TrainerConfig
    # Import the SFTTrainer - adjust path as needed
    try:
        from myllm.Train.sft_trainer import SFTTrainer
    except ImportError:
        # If the module path is different, adjust accordingly
        from .sft_trainer import SFTTrainer

    # ============================================================
    # SFT Collate Function
    # ============================================================
    def sft_collate_fn(batch, tokenizer):
        """
        Combines multiple SFT dataset items into a batch with proper padding.

        Args:
            batch (list of dict): Each item has {"input_ids", "labels", "text", "instruction", "response"}
            tokenizer: Tokenizer for handling padding.

        Returns:
            dict: A batch dictionary containing:
                - input_ids
                - attention_mask
                - labels (will be processed by SFTTrainer.create_sft_labels)
        """
        texts = [item["text"] for item in batch]

        # Tokenize and pad to the longest sequence in the batch
        encoded = tokenizer.batch_encode(texts, padding=True, return_tensors="pt")

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # For SFT, labels will be processed by the trainer to mask instructions
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "instructions": [item["instruction"] for item in batch],
            "responses": [item["response"] for item in batch]
        }

    # ============================================================
    # SFT Dataset
    # ============================================================
    class SFTDataset(Dataset):
        """
        A dataset for SFT training that handles instruction-response pairs.
        """
        def __init__(self, instruction_response_pairs, tokenizer, response_template="### Response:"):
            self.pairs = instruction_response_pairs
            self.tokenizer = tokenizer
            self.response_template = response_template

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            """
            Return:
                dict with:
                - "input_ids": tokenized full text tensor
                - "labels": same as input_ids (will be masked by trainer)
                - "text": formatted instruction + response
                - "instruction": raw instruction text
                - "response": raw response text
            """
            instruction, response = self.pairs[idx]
            
            # Format the text with instruction and response
            formatted_text = f"### Instruction:\n{instruction}\n\n{self.response_template}\n{response}"
            
            # Tokenize the full sequence
            enc = self.tokenizer.encode(formatted_text, return_tensors="pt").squeeze(0)
            
            return {
                "input_ids": enc,
                "labels": enc,  # Will be processed by SFTTrainer
                "text": formatted_text,
                "instruction": instruction,
                "response": response
            }

    # ============================================================
    # SFT Trainer Configuration
    # ============================================================
    trainer_config = TrainerConfig(
        model_config_name="gpt2-small",
        tokenizer_name="gpt2",
        output_dir="./sft_output",
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        wandb_project="my-sft-training",
        wandb_run_name="gpt2-small-sft-experiment",
        wandb_tags=["sft", "instruction-tuning", "gpt2"],
        report_to=["wandb"],
        logging_steps=5,
        eval_steps=25,
        save_steps=50,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        max_grad_norm=1.0
    )
    
    # ============================================================
    # Add SFT-specific configurations after creation
    # ============================================================
    trainer_config.mask_instructions = True  # Only compute loss on response tokens
    trainer_config.response_template = "### Response:"

    # ============================================================
    # Load Model Config
    # ============================================================
    model_config = ModelConfig.from_name("gpt2-small")
    model_config.learning_rate = 1e-4  # Slightly lower LR for fine-tuning
    model_config.dropout = 0.1

    # ============================================================
    # Initialize SFT Trainer
    # ============================================================
    sft_trainer = SFTTrainer(trainer_config, model_config=model_config)

    # Setup model & tokenizer
    sft_trainer.setup_model()

    # Move model to device
    sft_trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft_trainer.model.to(sft_trainer.device)

    # ============================================================
    # ✅ Apply torch.compile for performance optimization
    # ============================================================
    print("Compiling model with torch.compile() for optimized performance...")
    sft_trainer.model = torch.compile(sft_trainer.model, backend="inductor", mode="default")

    # Setup optimizer (must be after moving model to device and compiling)
    sft_trainer.setup_optimizer()

    # ============================================================
    # Sample SFT Training Data (Instruction-Response Pairs)
    # ============================================================
    instruction_response_pairs = [
        (
            "What is the capital of France?",
            "The capital of France is Paris. It is located in the north-central part of the country."
        ),
        (
            "Explain photosynthesis in simple terms.",
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
        ),
        (
            "How do you make a paper airplane?",
            "To make a paper airplane: 1) Take a sheet of paper and fold it in half lengthwise. 2) Unfold it and fold the top corners into the center crease. 3) Fold the angled edges into the center crease again. 4) Fold the plane in half along the center crease. 5) Create the wings by folding each side down to align with the bottom of the plane."
        ),
        (
            "What are the benefits of exercise?",
            "Exercise has many benefits including: improved cardiovascular health, stronger muscles and bones, better mental health, weight management, increased energy levels, and better sleep quality."
        ),
        (
            "Translate 'Hello, how are you?' to Spanish.",
            "The Spanish translation of 'Hello, how are you?' is 'Hola, ¿cómo estás?'"
        ),
        (
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions."
        ),
        (
            "How do you cook pasta?",
            "To cook pasta: 1) Bring a large pot of salted water to a boil. 2) Add the pasta and stir occasionally. 3) Cook according to package directions until al dente. 4) Drain the pasta, reserving some pasta water if needed for sauce. 5) Serve immediately with your desired sauce."
        ),
        (
            "What causes seasons on Earth?",
            "Seasons are caused by Earth's tilted axis as it orbits the sun. When your hemisphere tilts toward the sun, it receives more direct sunlight and experiences summer. When it tilts away, it receives less direct sunlight and experiences winter."
        ),
        (
            "Explain the water cycle.",
            "The water cycle describes how water moves through Earth's systems: 1) Evaporation - water turns from liquid to gas. 2) Condensation - water vapor forms clouds. 3) Precipitation - water falls as rain, snow, etc. 4) Collection - water gathers in oceans, lakes, and rivers, then the cycle repeats."
        ),
        (
            "What is the difference between weather and climate?",
            "Weather refers to short-term atmospheric conditions in a specific place and time, like today's temperature and rainfall. Climate refers to long-term weather patterns averaged over many years in a particular region."
        )
    ]

    # ============================================================
    # Create SFT Dataset & DataLoader
    # ============================================================
    sft_dataset = SFTDataset(
        instruction_response_pairs, 
        tokenizer=sft_trainer.tokenizer,
        response_template=trainer_config.response_template
    )

    train_loader = DataLoader(
        sft_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: sft_collate_fn(batch, tokenizer=sft_trainer.tokenizer)
    )

    # Use same data for evaluation in this example
    # In practice, you'd use a separate validation set
    sft_trainer.train_dataloader = train_loader
    sft_trainer.eval_dataloader = train_loader

    # ============================================================
    # Display Sample Data
    # ============================================================
    print("Sample SFT training examples:")
    for i, (instruction, response) in enumerate(instruction_response_pairs[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction: {instruction}")
        print(f"Response: {response}")
        print(f"Formatted: {sft_dataset[i]['text'][:200]}...")

    # ============================================================
    # Run SFT Training
    # ============================================================
    print(f"\nStarting SFT training with {len(instruction_response_pairs)} instruction-response pairs...")
    print(f"Instruction masking: {trainer_config.mask_instructions}")
    print(f"Response template: '{trainer_config.response_template}'")
    
    sft_trainer.train()

   
    # ============================================================
    # Final Output
    # ============================================================
    print(f"\nSFT Training completed!")
    print(f"Output directory: {trainer_config.output_dir}")
    print(f"Best model saved at: {sft_trainer.best_checkpoint_path}")
    
    if sft_trainer.wandb_url != "N/A":
        print(f"W&B Run: {sft_trainer.wandb_url}")