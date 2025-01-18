import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPTModel
from utils import download_and_load_gpt2, load_weights_into_gpt
import argparse
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Model configuration
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Quantization function
def quantize_tensor(tensor, num_bits=8):
    scale = torch.max(torch.abs(tensor))
    q_max = 2**num_bits - 1
    tensor_scaled = tensor / scale
    tensor_quantized = torch.round(tensor_scaled * q_max)
    tensor_quantized = tensor_quantized / q_max
    return tensor_quantized * scale

# LoRA-enhanced linear layer
class LoraLinear(nn.Linear):
    def __init__(self, in_features, out_features, r=8, num_bits=8, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        self.lora_matrix_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_matrix_A = nn.Parameter(torch.randn(r, in_features))
        self.num_bits = num_bits
        self.weight.requires_grad = False

    def forward(self, x):
        qA = quantize_tensor(self.lora_matrix_A, num_bits=self.num_bits)
        qB = quantize_tensor(self.lora_matrix_B, num_bits=self.num_bits)
        lora_weights = torch.matmul(qB, qA)
        quantized_lora_weights = quantize_tensor(lora_weights, num_bits=self.num_bits)
        return super().forward(x) + F.linear(x, quantized_lora_weights)

# Replace layers with LoRA
def replace_with_qlora(model, rank=8, num_bits=4):
    for block in model.trf_blocks:  # Access the blocks in the model
        # Replace attention linear layers
        if isinstance(block.att.W_query, nn.Linear):
            block.att.W_query = LoraLinear(block.att.W_query.in_features, block.att.W_query.out_features, r=rank)
        if isinstance(block.att.W_key, nn.Linear):
            block.att.W_key = LoraLinear(block.att.W_key.in_features, block.att.W_key.out_features, r=rank)
        if isinstance(block.att.W_value, nn.Linear):
            block.att.W_value = LoraLinear(block.att.W_value.in_features, block.att.W_value.out_features, r=rank)
        if isinstance(block.att.out_proj, nn.Linear):
            block.att.out_proj = LoraLinear(block.att.out_proj.in_features, block.att.out_proj.out_features, r=rank)

        # Replace feedforward linear layers
        if isinstance(block.ff.layers[0], nn.Linear):
            block.ff.layers[0] = LoraLinear(block.ff.layers[0].in_features, block.ff.layers[0].out_features, r=rank)
        if isinstance(block.ff.layers[2], nn.Linear):
            block.ff.layers[2] = LoraLinear(block.ff.layers[2].in_features, block.ff.layers[2].out_features, r=rank)
    return model

# Parameter counting functions
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def count_lora_params(model):
    return sum(p.numel() for name, p in model.named_parameters() if 'lora_matrix' in name)

# Save the LoRA-enhanced model
def save_lora_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logging.info(f"LoRA-enhanced model saved to {path}")

# Load the LoRA-enhanced model
def load_lora_model(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"LoRA model file not found at {path}")
    model.load_state_dict(torch.load(path))
    logging.info(f"LoRA-enhanced model loaded from {path}")
    return model

def main(args):
    # Update configuration based on chosen model
    BASE_CONFIG.update(model_configs[args.model])
    logging.info("Model configuration completed.")
    
    # Load model
    model_size = args.model.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir=args.models_dir)
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    
    # Check if a pre-saved LoRA model exists and load it
    if args.load_model_path:
        model_lora = load_lora_model(model, args.load_model_path)
    else:
        # Apply LoRA
        model_lora = replace_with_qlora(model, rank=args.rank, num_bits=args.num_bits)
    
        # Save the modified model
        if args.save_model_path:
            save_lora_model(model_lora, args.save_model_path)

    # Calculate parameter statistics
    total_params = count_params(model_lora)
    lora_params = count_lora_params(model_lora)
    lora_percentage = (lora_params / total_params) * 100

    print(model_lora)
    print(f"Total Parameters: {total_params}")
    print(f"LoRA Parameters: {lora_params}")
    print(f"Percentage of LoRA Parameters: {lora_percentage:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT with LoRA and quantization.")
    parser.add_argument("--model", type=str, choices=list(model_configs.keys()), required=True, help="Model size (e.g., gpt2-small (124M)).")
    parser.add_argument("--models_dir", type=str, default="gpt2", help="Directory to store GPT-2 models.")
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA matrices.")
    parser.add_argument("--num_bits", type=int, default=4, help="Number of bits for quantization.")
    parser.add_argument("--save_model_path", type=str, default=None, help="Path to save the LoRA-enhanced model.")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to load a pre-saved LoRA-enhanced model.")
    args = parser.parse_args()
    main(args)
