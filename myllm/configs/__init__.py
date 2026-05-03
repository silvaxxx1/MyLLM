"""Lowercase alias for myllm.Configs — enables `from myllm.configs import ModelConfig`."""
from myllm.Configs import ModelConfig, GenerationConfig

__all__ = ["ModelConfig", "GenerationConfig"]
