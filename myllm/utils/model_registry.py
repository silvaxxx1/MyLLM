# ============================================
# model_registry.py - Unified Model Configuration
# ============================================
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ModelSpec:
    """Specification for a model variant"""
    url: str
    config_name: str  # Name in your ModelConfig registry
    expected_size: Optional[int] = None
    weight_mapper: Optional[str] = None  # Override default mapper if needed

@dataclass
class ModelFamily:
    """Configuration for a model family"""
    name: str
    variants: Dict[str, ModelSpec]
    default_mapper: str
    requires_auth: bool = False
    token_env_var: Optional[str] = None

# ============================================
# Model Registry - Easy to extend
# ============================================
MODEL_REGISTRY = {
    "gpt2": ModelFamily(
        name="gpt2",
        default_mapper="gpt2_mapper",
        variants={
            "gpt2-small": ModelSpec(
                url="https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors",
                config_name="gpt2-small",
                expected_size=500_000_000,
            ),
            "gpt2-medium": ModelSpec(
                url="https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors",
                config_name="gpt2-medium",
                expected_size=1_500_000_000,
            ),
            "gpt2-large": ModelSpec(
                url="https://huggingface.co/openai-community/gpt2-large/resolve/main/model.safetensors",
                config_name="gpt2-large",
                expected_size=3_000_000_000,
            ),
            "gpt2-xl": ModelSpec(
                url="https://huggingface.co/openai-community/gpt2-xl/resolve/main/model.safetensors",
                config_name="gpt2-xl",
                expected_size=6_000_000_000,
            ),
        }
    ),

    "llama2": ModelFamily(
        name="llama2",
        default_mapper="llama_mapper",
        requires_auth=True,
        token_env_var="HF_TOKEN",
        variants={
            "llama2-7b": ModelSpec(
                url="https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/model.safetensors",
                config_name="llama2-7b",
                expected_size=13_000_000_000,
            ),
            "llama2-13b": ModelSpec(
                url="https://huggingface.co/meta-llama/Llama-2-13b-hf/resolve/main/model.safetensors",
                config_name="llama2-13b",
                expected_size=26_000_000_000,
            ),
        }
    ),

    "llama3": ModelFamily(
        name="llama3",
        default_mapper="llama_mapper",
        requires_auth=True,
        token_env_var="HF_TOKEN",
        variants={
            "llama3-1b": ModelSpec(
                url="https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/model.safetensors",
                config_name="llama3-1b",
                expected_size=2_000_000_000,  # approximate, update when known
            ),
            "llama3-8b": ModelSpec(
                url="https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/model.safetensors",
                config_name="llama3-8b",
                expected_size=16_000_000_000,
            ),
        }
    ),

    "mistral": ModelFamily(
        name="mistral",
        default_mapper="mistral_mapper",
        variants={
            "mistral-7b-v0.1": ModelSpec(
                url="https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/model.safetensors",
                config_name="mistral-7b-v0.1",
                expected_size=14_000_000_000,
            ),
        }
    ),

    "phi": ModelFamily(
        name="phi",
        default_mapper="phi_mapper",
        variants={
            "phi-2": ModelSpec(
                url="https://huggingface.co/microsoft/phi-2/resolve/main/model.safetensors",
                config_name="phi-2",
                expected_size=5_500_000_000,
            ),
        }
    ),

    "gemma": ModelFamily(
        name="gemma",
        default_mapper="gemma_mapper",
        requires_auth=True,
        token_env_var="HF_TOKEN",
        variants={
            "gemma-2b": ModelSpec(
                url="https://huggingface.co/google/gemma-2b/resolve/main/model.safetensors",
                config_name="gemma-2b",
                expected_size=5_000_000_000,
            ),
            "gemma-7b": ModelSpec(
                url="https://huggingface.co/google/gemma-7b/resolve/main/model.safetensors",
                config_name="gemma-7b",
                expected_size=14_000_000_000,
            ),
        }
    ),
}
