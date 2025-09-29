# trainer/utils/model_utils.py
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def setup_model_compilation(model: torch.nn.Module, use_compile: bool, config) -> torch.nn.Module:
    """
    Handle model compilation with torch.compile if enabled
    
    Args:
        model: PyTorch model to compile
        use_compile: Whether to enable compilation
        config: Configuration object (for logging context)
        
    Returns:
        torch.nn.Module: Compiled model (or original if compilation fails/disabled)
    """
    if use_compile:
        try:
            logger.info("Compiling model with torch.compile()")
            model = torch.compile(model)
        except Exception as e:
            logger.error(f"torch.compile() failed: {e}")
    return model

def load_pretrained_weights(llm, pretrained_variant: Optional[str], model_family: str = "gpt2"):
    """
    Load pretrained weights into model if specified
    
    Args:
        llm: LLM instance with load method
        pretrained_variant: Pretrained model variant name
        model_family: Model family for loading
    """
    if pretrained_variant:
        logger.info(f"Loading pretrained weights: {pretrained_variant}")
        llm.load(model_variant=pretrained_variant, model_family=model_family)