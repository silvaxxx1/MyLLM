# ============================================================================
# myllm/tokenizers/factory.py
"""
Tokenizer factory for creating and managing tokenizer instances.
"""

from typing import Dict, Type, List, Any
import logging
import os

from .base import BaseTokenizer
from .gpt2_tokenizer import GPT2Tokenizer
from .llama2_tokenizer import LLaMA2Tokenizer
from .llama3_tokenizer import LLaMA3Tokenizer
from .trainable_tok import TrainableTokenizer  # <- custom trainable tokenizer

logger = logging.getLogger(__name__)

# Global registry for tokenizer classes
_TOKENIZER_REGISTRY: Dict[str, Type[BaseTokenizer]] = {}


def register_tokenizer(model_name: str, tokenizer_class: Type[BaseTokenizer]) -> None:
    """Register a tokenizer class dynamically."""
    if not issubclass(tokenizer_class, BaseTokenizer):
        raise TypeError(
            f"Tokenizer class must inherit from BaseTokenizer, got {tokenizer_class.__name__}"
        )
    _TOKENIZER_REGISTRY[model_name.lower().strip()] = tokenizer_class
    logger.info(f"Registered tokenizer {tokenizer_class.__name__} for model '{model_name}'")


def unregister_tokenizer(model_name: str) -> bool:
    """Unregister a tokenizer."""
    model_key = model_name.lower().strip()
    if model_key in _TOKENIZER_REGISTRY:
        del _TOKENIZER_REGISTRY[model_key]
        logger.info(f"Unregistered tokenizer for model '{model_name}'")
        return True
    return False


def get_tokenizer(model_name: str, **kwargs) -> BaseTokenizer:
    """
    Factory function to create tokenizer instances based on model name.
    """
    model_key = model_name.lower().strip()

    # Check registry first
    if model_key in _TOKENIZER_REGISTRY:
        tokenizer_class = _TOKENIZER_REGISTRY[model_key]
        return tokenizer_class(**kwargs)

    # Built-in tokenizers
    if model_key in ['gpt2', 'gpt-2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        return GPT2Tokenizer(model_name=model_key, **kwargs)

    elif model_key in ['llama2', 'llama-2', 'llama2-7b', 'llama2-13b', 'llama2-70b']:
        if 'model_path' not in kwargs:
            raise TypeError("LLaMA2 tokenizer requires 'model_path' argument pointing to .model file")
        model_path = kwargs['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model file not found: {model_path}")
        return LLaMA2Tokenizer(**kwargs)

    elif model_key in ['llama3', 'llama-3', 'llama3-8b', 'llama3-70b']:
        tokenizer_json_path = kwargs.get('tokenizer_json_path', None)
        if tokenizer_json_path and not os.path.exists(tokenizer_json_path):
            logger.warning(f"{tokenizer_json_path} not found, using default LLaMA3 config")
            kwargs.pop('tokenizer_json_path')
        return LLaMA3Tokenizer(**kwargs)

    elif model_key == 'trainable':
        # Only pass kwargs that TrainableTokenizer expects to avoid conflicts
        valid_kwargs = {k: v for k, v in kwargs.items() if k in ['vocab_size', 'min_frequency', 'special_tokens', 'model_name']}
        return TrainableTokenizer(**valid_kwargs)

    else:
        available_models = list_available_models()
        raise ValueError(
            f"Unsupported model '{model_name}'. Available models: {available_models}"
        )


def list_available_models() -> List[str]:
    """Return a sorted list of all available models."""
    built_in_models = [
        'gpt2', 'gpt-2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
        'llama2', 'llama-2', 'llama3', 'llama-3',
        'trainable'
    ]
    registered_models = list(_TOKENIZER_REGISTRY.keys())
    return sorted(set(built_in_models + registered_models))


def get_model_info(model_name: str) -> dict:
    """Return basic info about a tokenizer model."""
    try:
        tokenizer = get_tokenizer(model_name)
        return {
            "model_name": tokenizer.model_name,
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": tokenizer.special_tokens,
        }
    except Exception as e:
        return {"error": str(e)}
