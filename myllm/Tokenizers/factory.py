# ============================================================================
# myllm/tokenizers/factory.py
"""
Tokenizer factory for creating and managing tokenizer instances.
"""

from typing import Dict, Type, List, Any, Tuple, FrozenSet
import logging
import os

from myllm.Tokenizers.base import BaseTokenizer
from myllm.Tokenizers.gpt2_tokenizer import GPT2Tokenizer
from myllm.Tokenizers.llama2_tokenizer import LLaMA2Tokenizer
from myllm.Tokenizers.llama3_tokenizer import LLaMA3Tokenizer
from myllm.Tokenizers.trainable_tok import TrainableTokenizer
from myllm.Tokenizers.wrapper import TokenizerWrapper

logger = logging.getLogger(__name__)

# Registry for custom tokenizers
_TOKENIZER_REGISTRY: Dict[str, Type[BaseTokenizer]] = {}

# Cache for instantiated tokenizers (singleton pattern)
_INSTANCE_CACHE: Dict[Tuple[str, FrozenSet[Tuple[str, Any]]], TokenizerWrapper] = {}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def register_tokenizer(model_name: str, tokenizer_class: Type[BaseTokenizer]) -> None:
    """Register a tokenizer class dynamically."""
    if not issubclass(tokenizer_class, BaseTokenizer):
        raise TypeError(
            f"Tokenizer class must inherit from BaseTokenizer, got {tokenizer_class.__name__}"
        )
    _TOKENIZER_REGISTRY[model_name.lower().strip()] = tokenizer_class
    logger.info(f"✅ Registered tokenizer {tokenizer_class.__name__} for model '{model_name}'")


def unregister_tokenizer(model_name: str) -> bool:
    """Unregister a tokenizer."""
    model_key = model_name.lower().strip()
    if model_key in _TOKENIZER_REGISTRY:
        del _TOKENIZER_REGISTRY[model_key]
        logger.info(f"❌ Unregistered tokenizer for model '{model_name}'")
        return True
    return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_tokenizer(model_name: str, **kwargs) -> TokenizerWrapper:
    """
    Factory function to create tokenizer instances based on model name.
    Always returns a TokenizerWrapper for unified interface.
    Uses caching to avoid repeated instantiation.
    """
    model_key = model_name.lower().strip()
    cache_key = (model_key, frozenset(kwargs.items()))

    # Return from cache if exists
    if cache_key in _INSTANCE_CACHE:
        return _INSTANCE_CACHE[cache_key]

    # Registry lookup first
    if model_key in _TOKENIZER_REGISTRY:
        tokenizer_class = _TOKENIZER_REGISTRY[model_key]
        tokenizer = tokenizer_class(**kwargs)
        wrapper = TokenizerWrapper(tokenizer)
        _INSTANCE_CACHE[cache_key] = wrapper
        return wrapper

    # Built-in models (dict-driven lookup)
    BUILTIN_MODELS = {
        "gpt2": GPT2Tokenizer,
        "gpt-2": GPT2Tokenizer,
        "gpt2-medium": GPT2Tokenizer,
        "gpt2-large": GPT2Tokenizer,
        "gpt2-xl": GPT2Tokenizer,

        "llama2": LLaMA2Tokenizer,
        "llama-2": LLaMA2Tokenizer,
        "llama2-7b": LLaMA2Tokenizer,
        "llama2-13b": LLaMA2Tokenizer,
        "llama2-70b": LLaMA2Tokenizer,

        "llama3": LLaMA3Tokenizer,
        "llama-3": LLaMA3Tokenizer,
        "llama3-8b": LLaMA3Tokenizer,
        "llama3-70b": LLaMA3Tokenizer,

        "trainable": TrainableTokenizer,
    }

    if model_key in BUILTIN_MODELS:
        if model_key.startswith("llama2"):
            if "model_path" not in kwargs:
                raise TypeError("LLaMA2 tokenizer requires 'model_path' argument")
            if not os.path.exists(kwargs["model_path"]):
                raise FileNotFoundError(f"SentencePiece model file not found: {kwargs['model_path']}")

        elif model_key.startswith("llama3"):
            tokenizer_json_path = kwargs.get("tokenizer_json_path")
            if tokenizer_json_path and not os.path.exists(tokenizer_json_path):
                logger.warning(f"{tokenizer_json_path} not found, using default LLaMA3 config")
                kwargs.pop("tokenizer_json_path", None)

        elif model_key == "trainable":
            kwargs = {k: v for k, v in kwargs.items()
                      if k in ["vocab_size", "min_frequency", "special_tokens", "model_name"]}

        tokenizer = BUILTIN_MODELS[model_key](**kwargs)
        wrapper = TokenizerWrapper(tokenizer)
        _INSTANCE_CACHE[cache_key] = wrapper
        return wrapper

    # Unknown model → raise with available list
    raise ValueError(
        f"Unsupported model '{model_name}'. Available models: {list_available_models()}"
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def list_available_models() -> List[str]:
    """Return a sorted list of all available models."""
    built_in_models = [
        "gpt2", "gpt-2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "llama2", "llama-2", "llama3", "llama-3",
        "trainable",
    ]
    registered_models = list(_TOKENIZER_REGISTRY.keys())
    return sorted(set(built_in_models + registered_models))


def get_model_info(model_name: str) -> dict:
    """Return basic info about a tokenizer model."""
    try:
        tokenizer = get_tokenizer(model_name)
        return {
            "model_name": getattr(tokenizer.tokenizer, "model_name", model_name),
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": getattr(tokenizer.tokenizer, "special_tokens", None),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Debug / Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Single Tokenizer Test ===")
    tok = get_tokenizer("gpt2")
    print(tok)  # TokenizerWrapper representation

    text = "Hello, world!"
    ids = tok.encode(text)
    print("Encode:", ids)
    print("Decode:", tok.decode(ids))

    print("\n=== Batch Encode Test ===")
    batch = ["Hello world!", "This is a test sentence.", "Another one."]
    encoded = tok.batch_encode(batch)
    print("Input IDs:\n", encoded["input_ids"])
    print("Attention Mask:\n", encoded["attention_mask"])

    print("\n=== Caching Test ===")
    tok2 = get_tokenizer("gpt2")
    print("Same instance:", tok is tok2)

    print("\n=== Available Models ===")
    from myllm.Tokenizers.factory import list_available_models
    print(list_available_models())
