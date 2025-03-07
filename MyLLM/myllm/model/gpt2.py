import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration for the GPT-2 model (GPT_CONFIG_124) based on the original GPT-2 architecture.
GPT_CONFIG_124 = {
    "vocab_size": 50257,      # Size of the vocabulary (number of tokens)
    "context_length": 1024,   # Maximum length of the input context
    "emb_dim": 768,           # Dimension of the token and positional embeddings
    "n_head": 12,             # Number of attention heads in the multi-head attention mechanism
    "n_layer": 12,            # Number of transformer layers (blocks)
    "dropout": 0.1,           # Dropout rate for regularization
    "qkv_bias": False,        # Whether to include bias terms in the query, key, value projections
}

# Device handling for model training/inference on GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

class gpt2(nn.Module):
    """
    GPT-2 model implementation. A transformer-based language model for text generation.
    The model consists of token and positional embeddings, transformer blocks, layer normalization,
    and an output projection to predict the next token.
    """
    def __init__(self, config):
        """
        Initialize the GPT-2 model.
        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super(gpt2, self).__init__()

        # Token and position embeddings
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"]) 
        
        # Dropout layer for regularization
        self.drop = nn.Dropout(config["dropout"]) 
        
        # List of transformer blocks (stacked layers)
        self.trs_blk = nn.ModuleList([TransformerBlock(config) for _ in range(config['n_layer'])])
        
        # Layer normalization and final linear projection to vocab size
        self.norm = nn.LayerNorm(config["emb_dim"])
        self.proj = nn.Linear(config['emb_dim'], config['vocab_size']) 

    def forward(self, x):
        """
        Forward pass of the GPT-2 model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_len) representing token IDs.
        Returns:
            torch.Tensor: Output logits of shape (batch_size, sequence_len, vocab_size).
        """
        # Token embedding (batch_size, sequence_len) --> (batch_size, sequence_len, emb_dim)
        tok_emb = self.tok_emb(x)
        
        # Positional embedding (batch_size, sequence_len) --> (batch_size, sequence_len, emb_dim)
        pos_index = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos_index)
        
        # Add token and position embeddings together
        embedding = tok_emb + pos_emb
        
        # Apply dropout
        embedding = self.drop(embedding)
        
        # Pass through transformer blocks
        for block in self.trs_blk:
            embedding = block(embedding)  # Apply each transformer block
        
        # Normalize the output of the transformer
        normilized_output = self.norm(embedding)
        
        # Project the output back to the vocabulary size
        output = self.proj(normilized_output)

        return output

    
class TransformerBlock(nn.Module):
    """
    A single transformer block containing attention and feed-forward layers with residual connections.
    """
    def __init__(self, config):
        """
        Initialize the transformer block.
        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super().__init__()

        # Attention mechanism, layer normalization, dropout, and feed-forward network
        self.atten = FlashAttention(config,d_in=config["emb_dim"], d_out=config["emb_dim"])
        self.norm1 = nn.LayerNorm(config["emb_dim"])
        self.norm2 = nn.LayerNorm(config["emb_dim"])
        self.drop = nn.Dropout(config["dropout"]) 
        self.mlp = GPTMLP(config)

    def forward(self, x):
        """
        Forward pass through the transformer block.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_len, emb_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_len, emb_dim).
        """
        shortcut = x 
        
        # Apply the first layer normalization and attention
        x = self.norm1(x)
        x = self.atten(x)
        
        # Apply dropout and residual connection
        x = self.drop(x)
        x = shortcut + x

        shortcut = x
        
        # Apply the second layer normalization and MLP (feed-forward network)
        x = self.norm2(x)
        x = self.mlp(x)
        
        # Apply dropout and residual connection
        x = self.drop(x)
        x = shortcut + x

        return x

class FlashAttention(nn.Module):
    """
    FlashAttention mechanism for efficient multi-head self-attention.
    """
    def __init__(self, config, d_in, d_out, qkv_bias=False, drop=0.0):
        """
        Initialize the FlashAttention mechanism.
        Args:
            config (dict): Configuration dictionary containing model parameters.
            d_in (int): Input dimensionality.
            d_out (int): Output dimensionality (should be divisible by the number of heads).
            qkv_bias (bool): Whether to include biases in the Q, K, V projections.
            drop (float): Dropout rate for attention scores.
        """
        super().__init__()

        assert d_out % config["n_head"] == 0, "embed_dim is indivisible by num_heads"
        self.head_dim = d_out // config["n_head"] 
        self.d_out = d_out 
        self.qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.drop = drop

    def forward(self, x):
        """
        Forward pass through FlashAttention.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, emb_dim).
        """
        batch_size, seq_len, emb_dims = x.shape
        
        # Project input into Q, K, and V
        qkv = self.qkv(x)
        
        # Reshape Q, K, V to (batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, seq_len, 3, config["n_head"], self.head_dim)
        
        # Rearrange dimensions for multi-head attention
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        queries, keys, values = qkv 

        # Use dropout if the model is in training mode
        use_dropout = 0. if not self.training else self.drop

        # Perform FlashAttention (scaled dot-product attention)
        out = F.scaled_dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            attn_mask=None,
            dropout_p=use_dropout,
            is_causal=True
        )

        # Combine heads and reshape back to (batch_size, seq_len, d_out)
        context_vec = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)

        # Output projection
        context_vec = self.proj(context_vec)

        return context_vec

class GPTMLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) for use in GPT-2's feed-forward layers.
    """
    def __init__(self, config):
        """
        Initialize the GPT MLP.
        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super().__init__()

        # A sequential MLP consisting of a linear layer, GELU activation, and another linear layer
        self.layer = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),  # Projection to 4 times the embedding size
            nn.GELU(),  # GELU activation function
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),  # Back to embedding size
        )

    def forward(self, x):
        """
        Forward pass through the GPT MLP.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, emb_dim).
        """
        return self.layer(x)




# Use the provided GPT_CONFIG_124 or create a configuration dictionary
config = GPT_CONFIG_124

# Initialize the model
model = gpt2(config).to(device)

# Example input: let's assume we are working with a vocabulary of 50257 tokens
# and the input is a sequence of token IDs
input_ids = torch.randint(0, config['vocab_size'], (1, config['context_length'])).to(device)  # Random input tokens

# Forward pass
output = model(input_ids)

# Check the output shape
print("Output shape:", output.shape)

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import torch

import os
import urllib.request

# import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination):
    # Send a GET request to download the file

    try:
        with urllib.request.urlopen(url) as response:
            # Get the total file size from headers, defaulting to 0 if not present
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return

            # Define the block size for reading the file
            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(url)  # Extract filename from URL
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                # Open the destination file in binary write mode
                with open(destination, "wb") as file:
                    # Read the file in chunks and write to destination
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # Update progress bar
    except urllib.error.HTTPError:
        s = (
            f"The specified URL ({url}) is incorrect, the internet connection cannot be established,"
            "\nor the requested file is temporarily unavailable.\nPlease visit the following website"
            " for help: https://github.com/rasbt/LLMs-from-scratch/discussions/273")
        print(s)


# Alternative way using `requests`
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])



## Choose the model to use
CHOOSE_MODEL = "gpt2-medium (355M)"

## Base configuration settings for the model
BASE_CONFIG = {
    "vocab_size": 50257,     # Size of the vocabulary used by the model
    "context_length": 1024,  # Maximum context length the model can handle
    "drop_rate": 0.0,        # Dropout rate for regularization
    "qkv_bias": True         # Whether to use bias terms in query, key, and value projections
}

## Dictionary containing configurations for different GPT-2 model sizes
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},      # Config for small model
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},    # Config for medium model
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},     # Config for large model
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},       # Config for extra-large model
}

## Update the BASE_CONFIG with parameters specific to the chosen model
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = gpt2(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()