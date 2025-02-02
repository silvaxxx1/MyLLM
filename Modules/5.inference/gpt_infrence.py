import sys
import os
import logging
import argparse
import torch

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from UTILS.generate import TextGenerator  # Import TextGenerator class
from configs.gpt_config import GPT_CONFIG_124M, GPT_CONFIG_355M, model_names  # Import configurations

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Map model names to configuration and checkpoint paths
MODEL_CONFIG_MAP = {
    "gpt2-small (124M)": (GPT_CONFIG_124M, r"C:\Users\user\Documents\SILVA AI ROADMAP\MyLLM\inference\gpt_infrence\checkpoints\models\gpt2-small (124M)_model.pt"),
    "gpt2-medium (355M)": (GPT_CONFIG_355M, r"C:\Users\user\Documents\SILVA AI ROADMAP\MyLLM\inference\gpt_infrence\checkpoints\models\gpt2-medium (355M)_model.pt"),
    # Add paths for other models if needed
}

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained GPT model.")
    parser.add_argument('--prompt', type=str, required=True, help="Prompt to generate text from.")
    parser.add_argument('--max_length', type=int, default=100, help="Maximum number of tokens to generate.")
    parser.add_argument('--temperature', type=float, default=0.7, help="Temperature for controlling randomness.")
    parser.add_argument('--top_k', type=int, default=None, help="Top-K sampling (set to None for greedy decoding).")
    parser.add_argument('--sampling', type=str, choices=["greedy", "top_k", "nucleus"], default="greedy", help="Sampling method to use.")
    parser.add_argument('--model_name', type=str, choices=MODEL_CONFIG_MAP.keys(), required=True, help="Which GPT model to use.")
    
    args = parser.parse_args()

    # Select model configuration and checkpoint path
    config, model_path = MODEL_CONFIG_MAP[args.model_name]

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize text generator
    logging.info(f"Loading model: {args.model_name} from {model_path}")
    generator = TextGenerator(model_name="gpt2", device=device)

    # Generate text
    logging.info(f"Generating text with sampling method: {args.sampling}")
    generated_text = generator.generate(
        prompt=args.prompt,
        length=args.max_length,
        beams=5 if args.sampling == "greedy" else 1,  # Use beams for greedy search
        sampling=args.sampling,
        temperature=args.temperature
    )

    # Output the generated text
    print("\nGenerated Text:\n" + generated_text)
