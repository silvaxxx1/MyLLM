import os
import urllib.request
import torch
from safetensors.torch import load_file
from tqdm import tqdm
import sys
import time
from threading import Thread


# Spinner class for loading weights
class Spinner:
    def __init__(self, msg="Loading..."):
        self.msg = msg
        self.done = False

    def spinner_task(self):
        while not self.done:
            for ch in "|/-\\":
                print(f"\r{self.msg} {ch}", end="", flush=True)
                time.sleep(0.1)

    def __enter__(self):
        self.thread = Thread(target=self.spinner_task)
        self.thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done = True
        self.thread.join()
        print("\r" + " " * (len(self.msg) + 2) + "\r", end="")


def download_safetensors(model_name: str, model_dir: str, url: str) -> str:
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, model_name)
    if os.path.exists(filepath):
        print(f"File {filepath} already exists, skipping download.")
        return filepath

    print(f"Downloading {model_name} from {url} ...")

    # Download with tqdm progress bar
    with urllib.request.urlopen(url) as response:
        total_size = int(response.getheader('Content-Length').strip())
        chunk_size = 1024 * 1024  # 1MB
        with open(filepath, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=model_name, ascii=True
        ) as pbar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

    print("Download complete.")
    return filepath


def load_safetensors(filepath: str) -> dict:
    print(f"Loading weights from {filepath} ...")
    return load_file(filepath)


def load_gpt2_weights(model, params):
    num_blocks = len([k for k in params.keys() if k.startswith("h.") and ".attn.c_attn.weight" in k])

    # Loading embeddings and positional embeddings
    print("Loading embeddings and positional embeddings...")
    model.wte.weight.data.copy_(params["wte.weight"].detach().clone())
    model.wpe.weight.data.copy_(params["wpe.weight"].detach().clone())

    # Use tqdm to track block loading
    with tqdm(total=num_blocks, desc="Loading transformer blocks", ascii=True) as pbar:
        for i in range(num_blocks):
            prefix = f"h.{i}"

            c_attn_w = params[f"{prefix}.attn.c_attn.weight"].T
            c_attn_b = params.get(f"{prefix}.attn.c_attn.bias", None)
            model.transformer[f"block_{i}"].attn.qkv.weight.data.copy_(c_attn_w)
            if model.transformer[f"block_{i}"].attn.qkv.bias is not None and c_attn_b is not None:
                model.transformer[f"block_{i}"].attn.qkv.bias.data.copy_(c_attn_b)

            c_proj_w = params[f"{prefix}.attn.c_proj.weight"].T
            c_proj_b = params.get(f"{prefix}.attn.c_proj.bias", None)
            model.transformer[f"block_{i}"].attn.proj.weight.data.copy_(c_proj_w)
            if model.transformer[f"block_{i}"].attn.proj.bias is not None and c_proj_b is not None:
                model.transformer[f"block_{i}"].attn.proj.bias.data.copy_(c_proj_b)

            c_fc_w = params[f"{prefix}.mlp.c_fc.weight"].T
            c_fc_b = params.get(f"{prefix}.mlp.c_fc.bias", None)
            c_proj_w_mlp = params[f"{prefix}.mlp.c_proj.weight"].T
            c_proj_b_mlp = params.get(f"{prefix}.mlp.c_proj.bias", None)

            model.transformer[f"block_{i}"].mlp.fc.weight.data.copy_(c_fc_w)
            if model.transformer[f"block_{i}"].mlp.fc.bias is not None and c_fc_b is not None:
                model.transformer[f"block_{i}"].mlp.fc.bias.data.copy_(c_fc_b)

            model.transformer[f"block_{i}"].mlp.proj.weight.data.copy_(c_proj_w_mlp)
            if model.transformer[f"block_{i}"].mlp.proj.bias is not None and c_proj_b_mlp is not None:
                model.transformer[f"block_{i}"].mlp.proj.bias.data.copy_(c_proj_b_mlp)

            ln_1_w = params[f"{prefix}.ln_1.weight"]
            ln_1_b = params[f"{prefix}.ln_1.bias"]
            ln_2_w = params[f"{prefix}.ln_2.weight"]
            ln_2_b = params[f"{prefix}.ln_2.bias"]

            model.transformer[f"block_{i}"].norm1.weight.data.copy_(ln_1_w)
            model.transformer[f"block_{i}"].norm1.bias.data.copy_(ln_1_b)
            model.transformer[f"block_{i}"].norm2.weight.data.copy_(ln_2_w)
            model.transformer[f"block_{i}"].norm2.bias.data.copy_(ln_2_b)

            pbar.update(1)

    print("Loading final LayerNorm and lm_head...")
    model.ln_f.weight.data.copy_(params["ln_f.weight"])
    model.ln_f.bias.data.copy_(params["ln_f.bias"])

    if "lm_head.weight" in params:
        model.lm_head.weight.data.copy_(params["lm_head.weight"])
    else:
        model.lm_head.weight = model.wte.weight

def get_gpt2_safetensors_url(model_variant: str) -> str:
    URL_DIR = {
        "gpt2-small": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
        "gpt2-xl": "gpt2-xl"
    }
    if model_variant not in URL_DIR:
        raise ValueError(f"Unknown GPT2 variant {model_variant}")
    base_url = "https://huggingface.co/openai-community"
    return f"{base_url}/{URL_DIR[model_variant]}/resolve/main/model.safetensors"
