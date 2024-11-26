import argparse
import logging
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from Llama_configs import LLAMA32_CONFIG
from Llama3 import Llama3
from load_Llama3 import load_weights_into_llama
from Llama3_tokenizer import Tokenizer

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Reduce the context length if necessary
old_context_length = LLAMA32_CONFIG["context_length"]
LLAMA32_CONFIG["context_length"] = 8192  # Setting the new context length

def rescale_theta(theta_old, context_length_old, context_length_new):
    """
    Rescale the RoPE (rotary positional encoding) parameters to match the new context length.
    """
    scaling_factor = context_length_new / context_length_old
    theta_new = theta_old * scaling_factor
    return theta_new

# Rescale the RoPE theta value based on the new context length
LLAMA32_CONFIG["rope_base"] = rescale_theta(
    LLAMA32_CONFIG["rope_base"],
    old_context_length,
    LLAMA32_CONFIG["context_length"]
)

logging.info(f"New RoPE theta: {LLAMA32_CONFIG['rope_base']}")

def calculate_model_memory(model, input_dtype=torch.float32):
    """
    Calculate the memory size of the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_buffers = sum(buf.numel() for buf in model.buffers())
    element_size = torch.tensor(0, dtype=input_dtype).element_size()

    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    return total_memory_gb

def load_model_weights(model, size_str):
    """
    Load the weights for the model based on its size.
    """
    combined_weights = {}
    if size_str == "1B":
        weights_file = hf_hub_download(
            repo_id=f"meta-llama/Llama-3.2-{size_str}-Instruct",
            filename="model.safetensors",
            local_dir=f"Llama-3.2-{size_str}-Instruct"
        )
        combined_weights = load_file(weights_file)
    else:
        for i in range(1, 3):  # Assuming weights are split into two parts
            weights_file = hf_hub_download(
                repo_id=f"meta-llama/Llama-3.2-{size_str}-Instruct",
                filename=f"model-0000{i}-of-00002.safetensors",
                local_dir=f"Llama-3.2-{size_str}-Instruct"
            )
            combined_weights.update(load_file(weights_file))
    
    load_weights_into_llama(model, LLAMA32_CONFIG, combined_weights)
    del combined_weights  # Free memory
    logging.info("Weights loaded successfully.")

def main(args):
    logging.info("Downloading tokenizer model...")
    tokenizer_file_path = hf_hub_download(
        repo_id=f"meta-llama/Llama-3.2-{args.size}-Instruct",
        filename="original/tokenizer.model",
        local_dir=f"Llama-3.2-{args.size}-Instruct"
    )

    logging.info("Initializing tokenizer...")
    tokenizer = Tokenizer(tokenizer_file_path)
    logging.info("Tokenizer initialized.")

    logging.info("Initializing model...")
    model = Llama3(LLAMA32_CONFIG)
    model.to(device)
    logging.info("Model initialized with configuration:")
    logging.info(LLAMA32_CONFIG)

    # Calculate and log the model's parameter count and memory usage
    total_params = sum(p.numel() for p in model.parameters())
    total_params_normalized = total_params - model.token_embedding.weight.numel()
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Total unique parameters (weight tying accounted): {total_params_normalized:,}")

    logging.info("Calculating memory usage...")
    logging.info(f"float32 memory usage: {calculate_model_memory(model, input_dtype=torch.float32):.2f} GB")
    logging.info(f"bfloat16 memory usage: {calculate_model_memory(model, input_dtype=torch.bfloat16):.2f} GB")

    # Print model architecture
    logging.info("Model Architecture:")
    print(model)  # This will print the architecture to the console

    # Load model weights after printing the architecture
    logging.info("Loading model weights...")
    load_model_weights(model, args.size)

    logging.info("Model ready for inference.")
    sample_input = "Hello, how are you?"
    encoded_input = tokenizer.model.encode(sample_input)
    logging.info(f"Sample input: {sample_input}")
    logging.info(f"Encoded output: {encoded_input}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaMA Model Loader")
    parser.add_argument(
        "--size",
        type=str,
        default="1B",
        choices=["1B", "3B"],
        help="Model size to load: '1B' or '3B'. Default is '1B'."
    )
    args = parser.parse_args()

    logging.info("Starting main program...")
    main(args)
