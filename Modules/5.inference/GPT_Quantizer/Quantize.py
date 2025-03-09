import argparse
import logging
import time
import torch
import tiktoken
from load_weights import download_and_load_gpt2, load_weights_into_gpt
from model import GPTModel
from quantizer import replace_module_and_quantize, W8A16LinearLayer
from generate import generate
from eval import calculate_perplexity  # Import the evaluation function

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Model configurations
MODEL_CONFIGS = {
    "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "size": "124M"},
    "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "size": "355M"},
    "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "size": "774M"},
    "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "size": "1558M"},
}

def setup_model(model_name: str):
    """Loads model weights and initializes the model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model name '{model_name}'. Choose from: {list(MODEL_CONFIGS.keys())}")

    model_size = MODEL_CONFIGS[model_name]["size"]
    logging.info(f"Downloading and loading weights for {model_name} ({model_size})...")
    
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    config.update(MODEL_CONFIGS[model_name])

    model = GPTModel(config)
    load_weights_into_gpt(model, params)
    model.eval()
    
    return model

def model_info(model):
    """Returns number of parameters and model size in MB."""
    num_params = sum(p.numel() for p in model.parameters())
    size_in_MB = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 * 1024)
    return num_params, size_in_MB

def run_inference(model, prompt, max_new_tokens, temperature, top_k):
    """Runs text generation with the model."""
    device = "cpu"
    model.to(device)
    tokenizer = tiktoken.get_encoding('gpt2')
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    start_time = time.time()
    generated_text = generate(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        context_size=512,
        tokenizer=tokenizer,
        temperature=temperature,
        top_k=top_k,
        top_p=None,
    )
    end_time = time.time()

    logging.info(f"Generated text: {generated_text}")
    logging.info(f"Inference Time: {end_time - start_time:.4f} seconds")

def compare_perplexity_scores(before, after):
    """Calculates and logs comparison metrics between perplexity scores."""
    absolute_diff = abs(before - after)
    relative_diff = (absolute_diff / before) * 100 if before != 0 else float('inf')
    ratio = after / before if before != 0 else float('inf')
    
    logging.info(f"Comparison of Perplexity Scores:")
    logging.info(f"Absolute Difference: {absolute_diff:.4f}")
    logging.info(f"Relative Difference: {relative_diff:.2f}%")
    logging.info(f"Perplexity Ratio (after/before): {ratio:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Automated GPT-2 model loading and inference")
    parser.add_argument("--model", type=str, choices=MODEL_CONFIGS.keys(), default="gpt2-medium",
                        help="Choose GPT-2 model size")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling")
    parser.add_argument("--quantize", action="store_true", help="Enable model quantization")
    parser.add_argument("--eval_text", type=str, default=None, help="Text for perplexity evaluation (optional)")

    args = parser.parse_args()

    logging.info(f"Loading model: {args.model}")
    model = setup_model(args.model)
    num_params, size_in_MB = model_info(model)

    logging.info(f"Model: {args.model}")
    logging.info(f"Number of parameters: {num_params}")
    logging.info(f"Size in MB: {size_in_MB:.2f} MB")
    logging.info(f"Size in GB: {size_in_MB / 1024:.4f} GB")

    # **Run inference on the original model**
    logging.info("Running inference on the original model...")
    run_inference(model, args.prompt, args.max_tokens, args.temperature, args.top_k)

    # **Perplexity Evaluation on input text (before quantization)**
    perplexity_before = None
    if args.eval_text:
        logging.info(f"Evaluating Perplexity on text (before quantization): '{args.eval_text}'")
        perplexity_before = calculate_perplexity(model, tiktoken.get_encoding("gpt2"), args.eval_text)
        logging.info(f"Perplexity Score (before quantization): {perplexity_before:.4f}")

    # **Quantize the model**
    if args.quantize:
        logging.info("Quantizing model...")
        replace_module_and_quantize(model, W8A16LinearLayer, ["out_head"])
        logging.info("Quantization complete.")

    # **Run inference on the quantized model**
    logging.info("Running inference on the quantized model...")
    run_inference(model, args.prompt, args.max_tokens, args.temperature, args.top_k)

    # **Perplexity Evaluation on input text (after quantization)**
    perplexity_after = None
    if args.eval_text:
        logging.info(f"Evaluating Perplexity on text (after quantization): '{args.eval_text}'")
        perplexity_after = calculate_perplexity(model, tiktoken.get_encoding("gpt2"), args.eval_text)
        logging.info(f"Perplexity Score (after quantization): {perplexity_after:.4f}")

    # **Comparison of the Perplexity Scores (before and after quantization)**
    if perplexity_before is not None and perplexity_after is not None:
        compare_perplexity_scores(perplexity_before, perplexity_after)

if __name__ == "__main__":
    main()
