# ============================================================================
# myllm/tokenizers/base.py
"""
Abstract base class for all tokenizers in the MyLLM system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseTokenizer(ABC):
    """
    Abstract base class defining the tokenizer interface for MyLLM.
    
    This class ensures all tokenizers provide a consistent API while allowing
    each implementation to use its native tokenization library for optimal
    compatibility and performance.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the base tokenizer.
        
        Args:
            **kwargs: Configuration parameters for the tokenizer
        """
        self._vocab_size: Optional[int] = None
        self._special_tokens: Dict[str, int] = {}
        self._reverse_special_tokens: Dict[int, str] = {}
        self._model_name: Optional[str] = kwargs.get('model_name')
        
    @abstractmethod
    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encode text into a list of token IDs.
        """
        pass
    
    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.
        """
        pass
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        if self._vocab_size is None:
            raise RuntimeError("Vocabulary size not initialized")
        return self._vocab_size
    
    @property
    def special_tokens(self) -> Dict[str, int]:
        """Get a copy of the special tokens mapping."""
        return self._special_tokens.copy()
    
    @property
    def model_name(self) -> Optional[str]:
        """Get the model name associated with this tokenizer."""
        return self._model_name

    @model_name.setter
    def model_name(self, value: Optional[str]):
        """Set the model name (will be lowercased if string)."""
        if isinstance(value, str):
            self._model_name = value.lower()
        else:
            self._model_name = value
    
    def get_special_token_id(self, token_name: str) -> Optional[int]:
        return self._special_tokens.get(token_name)
    
    def get_special_token_name(self, token_id: int) -> Optional[str]:
        return self._reverse_special_tokens.get(token_id)
    
    def is_special_token(self, token_id: int) -> bool:
        return token_id in self._reverse_special_tokens
    
    def _register_special_token(self, name: str, token_id: int) -> None:
        self._special_tokens[name] = token_id
        self._reverse_special_tokens[token_id] = name
    
    def _validate_text_input(self, text: Any) -> None:
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, got {type(text).__name__}")
    
    def _validate_ids_input(self, ids: Any) -> None:
        if not isinstance(ids, list):
            raise TypeError(f"Expected list input, got {type(ids).__name__}")
        if not all(isinstance(id_, int) for id_ in ids):
            raise TypeError("All token IDs must be integers")
    
    def __len__(self) -> int:
        return self.vocab_size
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self._vocab_size}, model='{self._model_name}')"
