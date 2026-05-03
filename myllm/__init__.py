"""
myllm — A from-scratch LLM framework covering the full pipeline:
tokenization → attention → training → RLHF → inference.

Quick start:
    from myllm import LLM, ModelConfig, GenerationConfig
    from myllm.train import SFTTrainer, SFTTrainerConfig
    from myllm.tokenizers import get_tokenizer
"""

# ── Core ──────────────────────────────────────────────────────────────────────
from .Configs import ModelConfig, GenerationConfig
from .api import LLM
from .model import GPT

# ── Trainers ──────────────────────────────────────────────────────────────────
from .Train import (
    SFTTrainer,
    DPOTrainer,
    PPOTrainer,
    PretrainTrainer,
)
from .Train.configs import (
    SFTTrainerConfig,
    DPOTrainerConfig,
    PPOTrainerConfig,
    TrainerConfig,
)

# ── Tokenizers ────────────────────────────────────────────────────────────────
from .Tokenizers import (
    get_tokenizer,
    GPT2Tokenizer,
    LLaMA2Tokenizer,
    LLaMA3Tokenizer,
)

# ── Submodule aliases (enables `from myllm.train import ...` style) ───────────
from . import Train as train
from . import Tokenizers as tokenizers
from . import Configs as configs
from . import utils

__version__ = "0.1.0"

__all__ = [
    # Core
    "LLM",
    "GPT",
    "ModelConfig",
    "GenerationConfig",
    # Trainers
    "SFTTrainer",
    "DPOTrainer",
    "PPOTrainer",
    "PretrainTrainer",
    # Trainer configs
    "SFTTrainerConfig",
    "DPOTrainerConfig",
    "PPOTrainerConfig",
    "TrainerConfig",
    # Tokenizers
    "get_tokenizer",
    "GPT2Tokenizer",
    "LLaMA2Tokenizer",
    "LLaMA3Tokenizer",
    # Submodules
    "train",
    "tokenizers",
    "configs",
    "utils",
    # Meta
    "__version__",
]
