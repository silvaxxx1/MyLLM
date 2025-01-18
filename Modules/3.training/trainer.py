import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse 
import torch 
import torch.nn as nn
import logging
from train import trainV1, eval_model
from data.dataloader import GPTDataLoader  
from models.GPT.GPT import GPTModel 
from configs.gpt_config import GPT_CONFIG_124M  # Import your configuration
from train_utils import plot_losses , save_model, load_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log the start of the training process
    logging.info("Starting training...")
    logging.info(f"Using device: {device}")
    logging.info(f"Training data path: {args.train_file}")
    logging.info(f"Validation data path: {args.val_file}")
    logging.info(f"Training for {args.epochs} epochs with a batch size of {args.batch_size}.")

    # Load the datasets using the GPTDataLoader
    train_data = GPTDataLoader(args.train_file, args.max_len, args.stride, args.batch_size)
    val_data = GPTDataLoader(args.val_file, args.max_len, args.stride, args.batch_size)

    # Create the model using the configuration from GPT_CONFIG_124M
    model = GPTModel(GPT_CONFIG_124M)  # Use the configuration

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Start training and log the process
    train_losses, val_losses, track_tokens_seen = trainV1(
        model,
        train_data,
        val_data,
        optimizer,
        device, 
        args.epochs,
        args.eval_freq,
        args.eval_iter,
        args.start_context,
        args.tokenizer
    )

    # Log the completion of training
    logging.info("Training completed.")

    # Save the model after each epoch
    save_model(model, optimizer, args.epochs, args.save_model)

    # Prepare for visualization
    epochs_tensor = torch.linspace(0, args.epochs, len(train_losses))

    plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model.")
    
   # Set default values for training and validation data file paths
    parser.add_argument('--train_file', type=str, 
                        default='C:\\Users\\user\\Documents\\SILVA AI ROADMAP\\MyLLM\\data\\train_ids.bin',
                        help='Path to the training data file.')
    parser.add_argument('--val_file', type=str, 
                        default='C:\\Users\\user\\Documents\\SILVA AI ROADMAP\\MyLLM\\data\\val_ids.bin',
                        help='Path to the validation data file.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum length of input sequences.')
    parser.add_argument('--stride', type=int, default=256, help='Stride for creating overlapping sequences.')
    parser.add_argument('--eval_freq', type=int, default=100, help='Frequency of evaluation (in steps).')
    parser.add_argument('--eval_iter', type=int, default=10, help='Number of batches to evaluate.')
    parser.add_argument('--start_context', type=str, default='', help='Initial context for text generation.')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='Tokenizer type to use.')
    parser.add_argument('--save_model', type=str, default='model_checkpoint.pth', help='Path to save the model.')
    parser.add_argument('--load_model', type=str, default='', help='Path to load the model checkpoint (if any).')

    args = parser.parse_args()
    main(args)
