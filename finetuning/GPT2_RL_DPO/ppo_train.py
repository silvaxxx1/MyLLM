import argparse
import logging
import torch
import time
from torch.utils.data import DataLoader
from utils.load_weights import download_and_load_gpt2, load_weights_into_gpt
from utils.finetune_utils import generate_and_print_sample
from data import PrefrenceDataset, load_data, split_data, download_data, custom_collate_fn, format_input
from utils.model import GPTModel
import tiktoken
from ppo_loss import evaluate_dpo_loss_loader, compute_dpo_loss_batch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model with DPO.")
    
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training.")
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of epochs for training.")
    parser.add_argument('--learning_rate', type=float, default=5e-6, help="Learning rate for AdamW optimizer.")
    parser.add_argument('--beta', type=float, default=0.1, help="Beta value for the DPO loss.")
    parser.add_argument('--eval_freq', type=int, default=5, help="Frequency of evaluations.")
    parser.add_argument('--eval_iter', type=int, default=5, help="Number of iterations for evaluation.")
    parser.add_argument('--model', type=str, required=True, help="Model size (e.g., gpt2-small).")
    parser.add_argument('--start_context', type=str, default=None, help="Starting context for text generation.")
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Download and load data
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    output_path = download_data(url)
    data = load_data(output_path)
    train_data, val_data, test_data = split_data(data)

    # Tokenizer initialization
    tokenizer = tiktoken.get_encoding('gpt2')
    train_dataset = PrefrenceDataset(train_data, tokenizer)
    val_dataset = PrefrenceDataset(val_data, tokenizer)

    # Load datasets into DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, drop_last=True)

    # Model configuration
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # Download and load model
    model_size = args.model.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    policy_model = GPTModel(BASE_CONFIG)
    ref_model = GPTModel(BASE_CONFIG)

    load_weights_into_gpt(policy_model, params)
    policy_model.eval()

    load_weights_into_gpt(ref_model, params)
    ref_model.eval()

    # Set random seed for reproducibility
    torch.manual_seed(123)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Start the training
    start_time = time.time()

    # Evaluate initial loss
    res = evaluate_dpo_loss_loader(
        policy_model=policy_model,
        reference_model=ref_model,
        train_loader=train_loader,
        val_loader=val_loader,
        beta=args.beta,
        eval_iter=args.eval_iter
    )

    logger.info(f"Training loss: {res['train_loss']:.3f}")
    logger.info(f"Validation loss: {res['val_loss']:.3f}")
    logger.info(f"Train reward margin: {res['train_chosen_reward'] - res['train_rejected_reward']:.3f}")
    logger.info(f"Val reward margin: {res['val_chosen_reward'] - res['val_rejected_reward']:.3f}")

    # Call the training function
    tracking = train_model_dpo_simple(
        policy_model=policy_model,
        reference_model=ref_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        beta=args.beta,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context=format_input(val_data[2]) if args.start_context is None else args.start_context,
        tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

def train_model_dpo_simple(
    policy_model, reference_model, train_loader, val_loader,
    optimizer, num_epochs, beta,
    eval_freq, eval_iter, start_context, tokenizer
):
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": []
    }
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        policy_model.train()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )

            loss.backward()
            optimizer.step()

            tokens_seen += batch["chosen"].numel()
            global_step += 1

            if global_step % eval_freq == 0:
                res = evaluate_dpo_loss_loader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    beta=beta,
                    eval_iter=eval_iter
                )
                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)

                logger.info(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margin {res['train_chosen_reward'] - res['train_rejected_reward']:.3f}, "
                    f"Val reward margin {res['val_chosen_reward'] - res['val_rejected_reward']:.3f}"
                )

        # Generate sample after each epoch
        generate_and_print_sample(
            model=policy_model,
            tokenizer=tokenizer,
            device=loss.device,
            start_context=start_context
        )

    return tracking

if __name__ == '__main__':
    main()
