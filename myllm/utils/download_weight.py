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


def load_gpt2_weights(model, params, device="cuda", efficient=True):
    print(f"{'Efficiently' if efficient else 'Bulk'} loading GPT-2 weights to {device}...")

    def copy(name, dest):
        tensor = params.get(name)
        if tensor is not None and dest is not None:
            dest.data.copy_(tensor.to(device) if efficient else tensor.clone().to(device))

    # Load embeddings
    copy("wte.weight", model.wte.weight)
    copy("wpe.weight", model.wpe.weight)

    num_blocks = len([k for k in params if k.startswith("h.") and ".attn.c_attn.weight" in k])
    with tqdm(total=num_blocks, desc="Loading transformer blocks", ascii=True) as pbar:
        for i in range(num_blocks):
            prefix = f"h.{i}"
            block = model.transformer[f"block_{i}"]

            copy(f"{prefix}.attn.c_attn.weight", block.attn.qkv.weight)
            copy(f"{prefix}.attn.c_attn.bias", block.attn.qkv.bias)
            copy(f"{prefix}.attn.c_proj.weight", block.attn.proj.weight)
            copy(f"{prefix}.attn.c_proj.bias", block.attn.proj.bias)
            copy(f"{prefix}.mlp.c_fc.weight", block.mlp.fc.weight)
            copy(f"{prefix}.mlp.c_fc.bias", block.mlp.fc.bias)
            copy(f"{prefix}.mlp.c_proj.weight", block.mlp.proj.weight)
            copy(f"{prefix}.mlp.c_proj.bias", block.mlp.proj.bias)
            copy(f"{prefix}.ln_1.weight", block.norm1.weight)
            copy(f"{prefix}.ln_1.bias", block.norm1.bias)
            copy(f"{prefix}.ln_2.weight", block.norm2.weight)
            copy(f"{prefix}.ln_2.bias", block.norm2.bias)

            pbar.update(1)

    print("Loading final LayerNorm and lm_head...")
    copy("ln_f.weight", model.ln_f.weight)
    copy("ln_f.bias", model.ln_f.bias)

    if hasattr(model, "lm_head"):
        if "lm_head.weight" in params:
            copy("lm_head.weight", model.lm_head.weight)
        else:
            model.lm_head.weight = model.wte.weight  # Tie weights fallback

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
