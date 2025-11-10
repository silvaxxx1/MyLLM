import os
import shutil
import torch
from safetensors.torch import load_file
from tqdm import tqdm
import time
import gc
import requests
from threading import Thread


# Spinner for UI feedback
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


# Utility to check free disk space (bytes)
def check_disk_space(dir_path, needed_bytes):
    try:
        total, used, free = shutil.disk_usage(dir_path)
        if free < needed_bytes:
            print(f"⚠️ Warning: Only {free / (1024**2):.2f} MB free on disk, but {needed_bytes / (1024**2):.2f} MB required.")
            return False
        return True
    except Exception as e:
        print(f"⚠️ Could not check disk space: {e}")
        return True  # Fail safe, proceed anyway


# -----------------------------
# Download + Verify .safetensors
# -----------------------------
def download_safetensors(model_name: str, model_dir: str, url: str, expected_size=None) -> str:
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, model_name)

    if os.path.exists(filepath):
        try:
            _ = load_file(filepath, device="cpu")
            print(f"✅ File {filepath} already exists and is valid.")
            return filepath
        except Exception:
            print(f"⚠️ Corrupted file at {filepath}, redownloading...")
            os.remove(filepath)

    if expected_size:
        if not check_disk_space(model_dir, expected_size):
            print("🛑 Not enough disk space to download model. Aborting.")
            raise RuntimeError("Insufficient disk space.")

    print(f"⬇️ Downloading {model_name} from {url} ...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True,
                                                 desc=model_name, ascii=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            _ = load_file(filepath, device="cpu")  # verify
            print("✅ Download complete and verified.")
            return filepath
        except Exception as e:
            print(f"❌ Download attempt {attempt + 1} failed: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            if attempt == max_retries - 1:
                raise RuntimeError("❌ Failed to download after retries.")
            print(f"⏳ Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)


# -----------------------------
# Load Weights (Safe Version) - 🆕 UPDATED
# -----------------------------
def load_safetensors(filepath: str, device: str = "cpu") -> dict:
    """
    Load safetensors file to specified device.
    
    Args:
        filepath: Path to .safetensors file
        device: Target device ("cpu", "cuda", etc.)
    
    Returns:
        Dictionary of tensors
    """
    print(f"📦 Loading weights from {filepath} to {device}...")
    try:
        params = load_file(filepath, device=device)
        print(f"✅ Loaded {len(params)} tensors")
        return params
    except Exception as e:
        raise RuntimeError(f"Failed to load safetensors: {str(e)}")


# -----------------------------
# Cleanup Utility - 🆕 ENHANCED
# -----------------------------
def cleanup(aggressive: bool = False):
    """
    Clean up memory.
    
    Args:
        aggressive: If True, performs more thorough cleanup (slower)
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.synchronize()
    if aggressive:
        time.sleep(0.5)


# -----------------------------
# Main Weight Loader - 🆕 DEPRECATED (use mapper classes instead)
# -----------------------------
def load_gpt2_weights_meta(model, config, params, device="cuda"):
    """
    DEPRECATED: Use GPT2WeightMapper instead.
    This is kept for backward compatibility.
    """
    print(f"⚠️ Warning: load_gpt2_weights_meta is deprecated. Use GPT2WeightMapper instead.")
    print(f"⚡ Loading GPT-2 weights safely to {device}")

    def safe_copy(name, dest):
        tensor = params.get(name)
        if tensor is not None and dest is not None:
            # transpose if shapes mismatch
            if tensor.shape[::-1] == dest.shape:
                tensor = tensor.T
            tensor = tensor.to(dest.device, non_blocking=True)
            with torch.no_grad():
                dest.copy_(tensor)

    # Move model to CPU for safe copying
    model.to("cpu")

    # Load embeddings
    safe_copy("wte.weight", model.wte.weight)
    if hasattr(model, "wpe"):
        safe_copy("wpe.weight", model.wpe.weight)

    # Count number of blocks
    num_blocks = len([k for k in params if k.startswith("h.") and ".attn.c_attn.weight" in k])
    print(f"Detected {num_blocks} transformer blocks in weights.")

    with tqdm(total=num_blocks, desc="Loading transformer blocks", ascii=True) as pbar:
        for i in range(num_blocks):
            prefix = f"h.{i}"
            block = model.transformer[f"block_{i}"]

            # Load attention weights
            safe_copy(f"{prefix}.attn.c_attn.weight", block.attn.qkv.weight)
            if f"{prefix}.attn.c_attn.bias" in params:
                safe_copy(f"{prefix}.attn.c_attn.bias", block.attn.qkv.bias)

            safe_copy(f"{prefix}.attn.c_proj.weight", block.attn.proj.weight)
            if f"{prefix}.attn.c_proj.bias" in params:
                safe_copy(f"{prefix}.attn.c_proj.bias", block.attn.proj.bias)

            # Load MLP weights
            safe_copy(f"{prefix}.mlp.c_fc.weight", block.mlp.fc.weight)
            safe_copy(f"{prefix}.mlp.c_fc.bias", block.mlp.fc.bias)
            safe_copy(f"{prefix}.mlp.c_proj.weight", block.mlp.proj.weight)
            safe_copy(f"{prefix}.mlp.c_proj.bias", block.mlp.proj.bias)

            # Load LayerNorm
            safe_copy(f"{prefix}.ln_1.weight", block.norm1.weight)
            safe_copy(f"{prefix}.ln_1.bias", block.norm1.bias)
            safe_copy(f"{prefix}.ln_2.weight", block.norm2.weight)
            safe_copy(f"{prefix}.ln_2.bias", block.norm2.bias)

            pbar.update(1)

    # Load final LayerNorm and LM head
    print("🎯 Loading final LayerNorm and LM head...")
    safe_copy("ln_f.weight", model.ln_f.weight)
    safe_copy("ln_f.bias", model.ln_f.bias)

    if hasattr(model, "lm_head"):
        if "lm_head.weight" in params:
            safe_copy("lm_head.weight", model.lm_head.weight)
        else:
            model.lm_head.weight = model.wte.weight

    # Move model to target device
    try:
        model.to(device)
    except RuntimeError as e:
        print(f"⚠️ Warning: Failed to move model to {device} due to: {e}")
        print("Model remains on CPU.")

    cleanup(aggressive=True)
    return model


# -----------------------------
# 🆕 Memory-Efficient Weight Loading Utilities
# -----------------------------
def estimate_tensor_memory(params: dict, dtype: torch.dtype = torch.float32) -> dict:
    """
    Estimate memory usage of loaded tensors.
    
    Args:
        params: Dictionary of tensors
        dtype: Data type to estimate for
    
    Returns:
        Dictionary with memory statistics in GB
    """
    bytes_per_element = 4 if dtype == torch.float32 else 2  # float32 vs float16
    
    total_elements = sum(p.numel() for p in params.values())
    total_bytes = total_elements * bytes_per_element
    total_gb = total_bytes / (1024 ** 3)
    
    return {
        "total_tensors": len(params),
        "total_elements": total_elements,
        "total_bytes": total_bytes,
        "total_gb": total_gb
    }


def load_safetensors_chunked(filepath: str, device: str = "cpu", chunk_size: int = 100):
    """
    Load safetensors in chunks to reduce memory spikes.
    Useful for very large models.
    
    Args:
        filepath: Path to .safetensors file
        device: Target device
        chunk_size: Number of tensors to load at once
    
    Returns:
        Dictionary of tensors
    """
    from safetensors import safe_open
    
    print(f"📦 Loading weights in chunks from {filepath}...")
    params = {}
    
    with safe_open(filepath, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        total_keys = len(keys)
        
        with tqdm(total=total_keys, desc="Loading tensors", ascii=True) as pbar:
            for i in range(0, total_keys, chunk_size):
                chunk_keys = keys[i:i + chunk_size]
                for key in chunk_keys:
                    params[key] = f.get_tensor(key).to(device)
                    pbar.update(1)
                
                # Clean up memory after each chunk
                if i % (chunk_size * 5) == 0:
                    cleanup()
    
    print(f"✅ Loaded {len(params)} tensors")
    return params


# -----------------------------
# Helper: Get Download URL
# -----------------------------
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