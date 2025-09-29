# examples/check_memory.py
"""
Check available GPU memory before running tests
"""

import torch
import psutil
import GPUtil

def check_system_resources():
    print("🔍 System Resource Check:")
    
    # CPU memory
    cpu_mem = psutil.virtual_memory()
    print(f"  RAM: {cpu_mem.available / (1024**3):.1f} GB available")
    
    # GPU memory
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"  GPU {gpu.id}: {gpu.memoryFree:.1f} MB free / {gpu.memoryTotal:.1f} MB total")
        
        # Current PyTorch memory usage
        print(f"  PyTorch allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
        print(f"  PyTorch cached: {torch.cuda.memory_reserved() / (1024**2):.1f} MB")
    else:
        print("  No GPU available")

def cleanup_gpu_memory():
    """Force cleanup of GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ GPU memory cleaned")

if __name__ == "__main__":
    check_system_resources()
    cleanup_gpu_memory()
    check_system_resources()