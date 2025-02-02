import sys
import os
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import numpy as np
import torch
from transformers import GPT2Model
from UTILS.model import GPTModel
from configs.gpt_config import BASE_CONFIG, model_configs, model_names  # Import the configurations

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Load a GPT model with specified configurations.")
parser.add_argument('--model_name', type=str, default="gpt2-small (124M)",
                    choices=list(model_names.keys()),  # Using model names from model_names
                    help="Name of the model to load (default: gpt2-small (124M))")
parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                    help="Directory to save model checkpoints (default: checkpoints)")

args = parser.parse_args()

# Log the chosen model and checkpoint directory
logging.info(f"Selected model: {args.model_name}")
logging.info(f"Checkpoint directory: {args.checkpoint_dir}")

# Retrieve the model path and model-specific config using the selected model name
model_path = model_names[args.model_name]
model_config = model_configs[args.model_name]

# Update BASE_CONFIG with model-specific parameters
BASE_CONFIG.update(model_config)

# Log the complete configuration
logging.info(f"Model configuration: {BASE_CONFIG}")

# Load the pre-trained GPT model from transformers
gpt_hf = GPT2Model.from_pretrained(model_path, cache_dir=args.checkpoint_dir)
gpt_hf.eval()

# Helper function to check weight shapes and assign weights
def assign_check(left, right):
    if left.shape != right.shape:
        logging.error(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())

def load_weights(gpt, gpt_hf):
    d = gpt_hf.state_dict()
    logging.info("Loading weights...")

    gpt.pos_emb.weight = assign_check(gpt.pos_emb.weight, d["wpe.weight"])
    gpt.tok_emb.weight = assign_check(gpt.tok_emb.weight, d["wte.weight"])
    
    for b in range(BASE_CONFIG["n_layers"]):
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign_check(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign_check(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign_check(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign_check(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign_check(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign_check(gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign_check(gpt.trf_blocks[b].att.out_proj.weight, d[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign_check(gpt.trf_blocks[b].att.out_proj.bias, d[f"h.{b}.attn.c_proj.bias"])
        
        gpt.trf_blocks[b].ff.layers[0].weight = assign_check(gpt.trf_blocks[b].ff.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign_check(gpt.trf_blocks[b].ff.layers[0].bias, d[f"h.{b}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign_check(gpt.trf_blocks[b].ff.layers[2].weight, d[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign_check(gpt.trf_blocks[b].ff.layers[2].bias, d[f"h.{b}.mlp.c_proj.bias"])

        gpt.trf_blocks[b].norm1.scale = assign_check(gpt.trf_blocks[b].norm1.scale, d[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign_check(gpt.trf_blocks[b].norm1.shift, d[f"h.{b}.ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign_check(gpt.trf_blocks[b].norm2.scale, d[f"h.{b}.ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign_check(gpt.trf_blocks[b].norm2.shift, d[f"h.{b}.ln_2.bias"])

    gpt.final_norm.scale = assign_check(gpt.final_norm.scale, d[f"ln_f.weight"])
    gpt.final_norm.shift = assign_check(gpt.final_norm.shift, d[f"ln_f.bias"])
    gpt.out_head.weight = assign_check(gpt.out_head.weight, d["wte.weight"])

# Define the model saving function (architecture + weights)
def save_model(gpt, model_name, checkpoint_dir):
    # Create the subdirectory for models
    models_dir = os.path.join(checkpoint_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Define the model save path
    model_save_path = os.path.join(models_dir, f"{model_name}_model.pt")
    
    # Save the entire model (architecture + weights)
    torch.save(gpt, model_save_path)  # Save the entire model
    logging.info(f"Model saved successfully at {model_save_path}")

    # Optional: Save only state_dict if you prefer
    state_dict_path = os.path.join(models_dir, f"{model_name}_state_dict.pt")
    torch.save(gpt.state_dict(), state_dict_path)
    logging.info(f"Model weights saved successfully at {state_dict_path}")

# Load the model and move to device
gpt = GPTModel(BASE_CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_weights(gpt, gpt_hf)
gpt.to(device)  # Move the model to the selected device

# Log that the model is ready
logging.info("Model loaded successfully and moved to the device.")

# Save the entire model after loading the weights
save_model(gpt, args.model_name, args.checkpoint_dir)
