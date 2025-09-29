# trainer/utils/memory_utils.py
import torch
import gc

def cleanup_memory():
    """
    Clean up GPU and CPU memory
    
    This should be called between tests or training runs
    to prevent memory leaks and fragmentation.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()