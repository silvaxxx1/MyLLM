import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import logging
import argparse
import torch
import tiktoken

from training.train_utils import generate_text  # Import your generate function
from configs.gpt_config import GPT_CONFIG_124M, GPT_CONFIG_355M, model_names  # Import your configurations

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model from the .pt file
def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)  # Load the saved model
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to encode the input prompt
def encode_prompt(prompt, tokenizer):
    return tokenizer.encode(prompt)

# Function to decode the generated tokens back to text
def decode_tokens(token_ids, tokenizer):
    return tokenizer.decode(token_ids)

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Load a GPT model and generate text.")
    parser.add_argument('--prompt', type=str, required=True, help="Prompt to generate text from.")
    parser.add_argument('--max_length', type=int, default=100, help="Maximum number of tokens to generate.")
    parser.add_argument('--temperature', type=float, default=0.7, help="Temperature for controlling randomness.")
    parser.add_argument('--top_k', type=int, default=None, help="Top K tokens for sampling.")
    parser.add_argument('--model_name', type=str, choices=model_names.keys(), required=True, help="Name of the model to use.")
    
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Map model names to configuration and model paths
    model_config_map = {
        "gpt2-small (124M)": (GPT_CONFIG_124M, r"C:\Users\user\Documents\SILVA AI ROADMAP\MyLLM\inference\gpt_infrence\checkpoints\models\gpt2-small (124M)_model.pt"),
        "gpt2-medium (355M)": (GPT_CONFIG_355M, r"C:\Users\user\Documents\SILVA AI ROADMAP\MyLLM\inference\gpt_infrence\checkpoints\models\gpt2-medium (355M)_model.pt"),
        # Add paths for other models here
        # "gpt2-large (774M)": (GPT_CONFIG_774M, r"model_path_for_gpt2_large"),
        # "gpt2-xl (1558M)": (GPT_CONFIG_1558M, r"model_path_for_gpt2_xl"),
    }

    # Load model configuration and path based on command line argument
    config, model_save_path = model_config_map[args.model_name]

    # Load model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_save_path, device)

    # Tokenize input prompt
    input_ids = torch.tensor(encode_prompt(args.prompt, tokenizer)).unsqueeze(0).to(device)

    # Generate text using the provided generate_text function
    output_ids = generate_text(
        model=model,
        idx=input_ids,
        max_new_tokens=args.max_length,
        context_size=config['context_length'],  # Use context length from config
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=50256  # Use the EOS token ID for GPT-2
    )

    # Decode the generated tokens into text
    generated_text = decode_tokens(output_ids.squeeze().tolist(), tokenizer)

    # Output the generated text
    print(f"Generated Text:\n{generated_text}")
