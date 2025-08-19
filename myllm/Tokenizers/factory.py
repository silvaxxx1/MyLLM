

# ============================================================================
# myllm/tokenizers/factory.py
"""
Tokenizer factory for creating and managing tokenizer instances.
"""

from typing import Dict, Type, List, Any
import logging
from pathlib import Path

from .base import BaseTokenizer
from .gpt2_tokenizer import GPT2Tokenizer
from .llama2_tokenizer import LLaMA2Tokenizer
from .llama3_tokenizer import LLaMA3Tokenizer

logger = logging.getLogger(__name__)

# Global registry for tokenizer classes
_TOKENIZER_REGISTRY: Dict[str, Type[BaseTokenizer]] = {}


def register_tokenizer(model_name: str, tokenizer_class: Type[BaseTokenizer]) -> None:
    """
    Register a tokenizer class for a specific model name.
    
    This function allows dynamic registration of new tokenizer implementations
    without modifying the core factory code, enabling easy extension.
    
    Args:
        model_name: Model identifier (case-insensitive)
        tokenizer_class: Tokenizer class inheriting from BaseTokenizer
        
    Raises:
        TypeError: If tokenizer_class doesn't inherit from BaseTokenizer
    """
    if not issubclass(tokenizer_class, BaseTokenizer):
        raise TypeError(
            f"Tokenizer class must inherit from BaseTokenizer, got {tokenizer_class.__name__}"
        )
    
    model_key = model_name.lower().strip()
    _TOKENIZER_REGISTRY[model_key] = tokenizer_class
    
    logger.info(f"Registered tokenizer {tokenizer_class.__name__} for model '{model_name}'")


def unregister_tokenizer(model_name: str) -> bool:
    """
    Unregister a tokenizer for a specific model name.
    
    Args:
        model_name: Model identifier to unregister
        
    Returns:
        True if tokenizer was unregistered, False if not found
    """
    model_key = model_name.lower().strip()
    if model_key in _TOKENIZER_REGISTRY:
        del _TOKENIZER_REGISTRY[model_key]
        logger.info(f"Unregistered tokenizer for model '{model_name}'")
        return True
    return False


def get_tokenizer(model_name: str, **kwargs) -> BaseTokenizer:
    """
    Factory function to create tokenizer instances based on model name.
    
    This function handles the creation of appropriate tokenizer instances
    using the native libraries that each model family actually uses.
    
    Args:
        model_name: Model identifier (case-insensitive)
        **kwargs: Additional arguments passed to tokenizer constructor
        
    Returns:
        Initialized tokenizer instance
        
    Raises:
        ValueError: If model_name is not supported
        TypeError: If required arguments are missing for the tokenizer
        FileNotFoundError: If required model files are not found
        
    Examples:
        >>> # GPT-2 tokenizer using tiktoken
        >>> gpt2_tok = get_tokenizer('gpt2')
        >>> 
        >>> # LLaMA2 tokenizer using SentencePiece
        >>> llama2_tok = get_tokenizer('llama2', model_path='tokenizer.model')
        >>> 
        >>> # LLaMA3 tokenizer using tiktoken + tokenizer.json
        >>> llama3_tok = get_tokenizer('llama3', tokenizer_json_path='tokenizer.json')
    """
    model_key = model_name.lower().strip()
    
    # Check registry first
    if model_key in _TOKENIZER_REGISTRY:
        tokenizer_class = _TOKENIZER_REGISTRY[model_key]
        try:
            return tokenizer_class(model_name=model_name, **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize {tokenizer_class.__name__}: {e}")
            raise
    
    # Built-in model mappings with native library usage
    try:
        # GPT-2 family (using tiktoken)
        if model_key in ['gpt2', 'gpt-2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            return GPT2Tokenizer(model_name=model_key, **kwargs)
        
        # GPT-3 family (using tiktoken)  
        elif model_key in ['gpt3', 'gpt-3', 'text-davinci-003', 'davinci', 'curie', 'babbage', 'ada']:
            return GPT2Tokenizer(model_name=model_key, **kwargs)
        
        # GPT-3.5 family (using tiktoken)
        elif model_key in ['gpt-3.5-turbo', 'gpt3.5', 'chatgpt']:
            return GPT2Tokenizer(model_name='gpt-3.5-turbo', **kwargs)
        
        # GPT-4 family (using tiktoken)
        elif model_key in ['gpt4', 'gpt-4', 'gpt-4-turbo', 'gpt-4o']:
            return GPT2Tokenizer(model_name=model_key, **kwargs)
        
        # LLaMA family (using SentencePiece)
        elif model_key in ['llama', 'llama-7b', 'llama-13b', 'llama-30b', 'llama-65b']:
            if 'model_path' not in kwargs:
                raise TypeError("LLaMA tokenizer requires 'model_path' argument pointing to .model file")
            return LLaMA2Tokenizer(legacy=True, **kwargs)
        
        # LLaMA2 family (using SentencePiece)
        elif model_key in ['llama2', 'llama-2', 'llama2-7b', 'llama2-13b', 'llama2-70b']:
            if 'model_path' not in kwargs:
                raise TypeError("LLaMA2 tokenizer requires 'model_path' argument pointing to .model file")
            return LLaMA2Tokenizer(**kwargs)
        
        # LLaMA3 family (using tiktoken + tokenizer.json)
        elif model_key in ['llama3', 'llama-3', 'llama3-8b', 'llama3-70b']:
            return LLaMA3Tokenizer(**kwargs)
        
        # Code Llama (using SentencePiece like LLaMA2)
        elif model_key in ['code-llama', 'codellama']:
            if 'model_path' not in kwargs:
                raise TypeError("Code Llama tokenizer requires 'model_path' argument pointing to .model file")
            return LLaMA2Tokenizer(**kwargs)
        
        # Placeholder implementations for future models
        elif model_key in ['mistral', 'mistral-7b', 'mixtral', 'mixtral-8x7b']:
            raise NotImplementedError(
                "Mistral tokenizer not yet implemented. "
                "Will use SentencePiece when available."
            )
        
        elif model_key in ['pi', 'inflection-pi']:
            raise NotImplementedError(
                "Inflection Pi tokenizer not yet implemented. "
                "Will use custom implementation when available."
            )
        
        elif model_key in ['gemma', 'gemma-2b', 'gemma-7b']:
            raise NotImplementedError(
                "Google Gemma tokenizer not yet implemented. "
                "Will use SentencePiece when available."
            )
        
        elif model_key in ['claude', 'claude-3']:
            raise NotImplementedError(
                "Anthropic Claude tokenizer not available. "
                "Use official Anthropic API."
            )
        
        else:
            # Provide helpful error with available models
            available_models = list_available_models()
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Available models: {available_models}\n"
                f"To add support for a new model, use register_tokenizer()."
            )
            
    except NotImplementedError:
        raise
    except Exception as e:
        logger.error(f"Failed to create tokenizer for '{model_name}': {e}")
        raise


def list_available_models() -> List[str]:
    """
    Get a list of all available model names.
    
    Returns:
        Sorted list of supported model identifiers
    """
    # Built-in supported models
    built_in_models = [
        # GPT family
        'gpt2', 'gpt-2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
        'gpt3', 'gpt-3', 'text-davinci-003', 'gpt-3.5-turbo', 
        'gpt4', 'gpt-4', 'gpt-4-turbo', 'gpt-4o',
        
        # LLaMA family  
        'llama', 'llama2', 'llama-2', 'llama3', 'llama-3',
        'code-llama', 'codellama'
    ]
    
    # Add registered models
    registered_models = list(_TOKENIZER_REGISTRY.keys())
    
    # Combine and sort
    all_models = sorted(set(built_in_models + registered_models))
    return all_models


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model's tokenizer.
    
    Args:
        model_name: Model identifier
        
    Returns:
        Dictionary with model information
        
    Raises:
        ValueError: If model is not supported
    """
    model_key = model_name.lower().strip()
    
    # Model information mapping
    model_info = {
        # GPT family
        'gpt2': {
            'family': 'GPT',
            'library': 'tiktoken', 
            'vocab_size': 50257,
            'special_tokens': ['<|endoftext|>'],
            'required_args': []
        },
        'gpt-3.5-turbo': {
            'family': 'GPT',
            'library': 'tiktoken',
            'vocab_size': 100277, 
            'special_tokens': ['<|endoftext|>', '<|im_start|>', '<|im_end|>'],
            'required_args': []
        },
        'gpt-4': {
            'family': 'GPT', 
            'library': 'tiktoken',
            'vocab_size': 100277,
            'special_tokens': ['<|endoftext|>', '<|im_start|>', '<|im_end|>'],
            'required_args': []
        },
        
        # LLaMA family
        'llama2': {
            'family': 'LLaMA',
            'library': 'sentencepiece',
            'vocab_size': 32000,
            'special_tokens': ['<s>', '</s>', '<unk>'],
            'required_args': ['model_path']
        },
        'llama3': {
            'family': 'LLaMA',
            'library': 'tiktoken',
            'vocab_size': 128256,
            'special_tokens': ['<|begin_of_text|>', '<|end_of_text|>', '<|eot_id|>'],
            'required_args': []
        }
    }
    
    # Check for exact match first
    if model_key in model_info:
        info = model_info[model_key].copy()
        info['model_name'] = model_name
        return info
    
    # Check for family matches
    for key, info in model_info.items():
        if model_key.startswith(key) or key in model_key:
            result = info.copy()
            result['model_name'] = model_name
            return result
    
    # Check registry
    if model_key in _TOKENIZER_REGISTRY:
        tokenizer_class = _TOKENIZER_REGISTRY[model_key]
        return {
            'model_name': model_name,
            'family': 'Custom',
            'library': 'custom',
            'tokenizer_class': tokenizer_class.__name__,
            'vocab_size': 'unknown',
            'special_tokens': 'unknown',
            'required_args': 'unknown'
        }
     
    raise ValueError(f"No information available for model '{model_name}'")



# GPT-2 using tiktoken
gpt2 = get_tokenizer('gpt2')
tokens = gpt2.encode("Hello world!", bos=True, eos=True)

# LLaMA2 using SentencePiece  
llama2 = get_tokenizer('llama2', model_path='tokenizer.model')
tokens = llama2.encode("Hello world!", bos=True, eos=True)

# LLaMA3 using tiktoken + tokenizer.json
llama3 = get_tokenizer('llama3', tokenizer_json_path='tokenizer.json')
formatted = llama3.apply_chat_template([
    {"role": "user", "content": "Hello!"}
])
