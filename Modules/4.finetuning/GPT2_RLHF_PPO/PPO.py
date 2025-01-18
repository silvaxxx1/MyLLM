import os
import torch
from transformers import AutoTokenizer
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead  # Import a model wrapper with value head
from trl import PPOConfig, PPOTrainer
from trl.core import LengthSampler
from transformers import pipeline
from tqdm import tqdm
import logging
import wandb
from data import build_dataset  # Import the build_dataset function from data.py

# ---------------------------
# Setup Logging
# ---------------------------
def setup_logging(log_dir="logs", log_file="training.log"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_dir, log_file)), logging.StreamHandler()]
    )
    logging.info("Logging initialized.")

# ---------------------------
# Collator Function
# ---------------------------
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# ---------------------------
# Main Training Loop
# ---------------------------
if __name__ == "__main__":
    # Argument parser
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 model with PPO.")
    parser.add_argument("--tokenized_data_dir", type=str, default="C:/Users/WinDows/SILVA/MyLLM_101_from_scratch/finetuning/GPT2_RLHF_PPO/processed_data", help="Directory where tokenized data is stored.")
    parser.add_argument("--model_name", type=str, default="lvwerra/gpt2-imdb", help="Model name to fine-tune.")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="Learning rate for PPO.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_dir=args.log_dir)

    # Load the tokenized dataset using build_dataset from data.py
    logging.info("Loading tokenized dataset...")
    logging.info(f"Dataset directory: {args.tokenized_data_dir}")
    dataset = build_dataset(dataset_name="stanfordnlp/imdb", tokenizer_name="gpt2", min_review_length=200)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"Tokenized dataset loaded. Sample dataset: {dataset}")

    # Initialize PPO configuration and trainer
    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        log_with="wandb",  # Logging with Weights and Biases
    )

    # Initialize Weights and Biases
    wandb.init(project="PPO-Training", config=config)

    # Load the model and reference model for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)

    # Initialize PPOTrainer
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    # Sentiment analysis pipeline (reward model)
    device = ppo_trainer.accelerator.device
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

    # Reward computation (positive/negative sentiment)
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
    
    logging.info("Example sentiment analysis:")
    text = "this movie was really bad!!"
    sentiment_result_1 = sentiment_pipe(text, **sent_kwargs)
    logging.info(f"Sentiment for 'bad' movie text: {sentiment_result_1}")
    
    # Logging message indicating printing of a sample reward
    logging.info("Printing a sample of the reward computation:")
    text = "this movie was really good!!"
    sentiment_result_2 = sentiment_pipe(text, **sent_kwargs)
    logging.info(f"Sentiment for 'good' movie text: {sentiment_result_2}")

    # Generation configuration
    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    response_generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # ---------------------------
    # Log message indicating start of PPO alignment
    # ---------------------------
    logging.info("Starting PPO alignment...")  # Log message for starting PPO alignment

    # Training loop
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        #### Phase 1: Get trajectories from the offline policy
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            response_generation_kwargs["max_new_tokens"] = gen_len  # Number of tokens to generate (chosen randomly)
            response = ppo_trainer.generate(query, **response_generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])  # Only keep generated tokens

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Phase 2: Compute rewards
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        #### PPO update
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        ppo_trainer.log_stats(stats, batch, rewards)

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("gpt2-imdb-pos-v2", push_to_hub=False)
    tokenizer.save_pretrained("gpt2-imdb-pos-v2", push_to_hub=False)

    logging.info("Training complete and model saved.")
