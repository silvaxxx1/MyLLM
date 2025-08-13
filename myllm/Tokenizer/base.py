

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
        
        Args:
            text: Input text to tokenize
            bos: Whether to prepend beginning-of-sequence token
            eos: Whether to append end-of-sequence token
            
        Returns:
            List of integer token IDs
            
        Raises:
            TypeError: If text is not a string
            ValueError: If text cannot be encoded
        """
        pass
    
    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.
        
        Args:
            ids: List of token IDs to decode
            
        Returns:
            Decoded text string
            
        Raises:
            TypeError: If ids is not a list of integers
            ValueError: If token IDs cannot be decoded
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
    
    def get_special_token_id(self, token_name: str) -> Optional[int]:
        """
        Get the token ID for a special token by name.
        
        Args:
            token_name: Name of the special token (e.g., 'bos', 'eos')
            
        Returns:
            Token ID if found, None otherwise
        """
        return self._special_tokens.get(token_name)
    
    def get_special_token_name(self, token_id: int) -> Optional[str]:
        """
        Get the special token name for a token ID.
        
        Args:
            token_id: Token ID to look up
            
        Returns:
            Token name if found, None otherwise
        """
        return self._reverse_special_tokens.get(token_id)
    
    def is_special_token(self, token_id: int) -> bool:
        """
        Check if a token ID represents a special token.
        
        Args:
            token_id: Token ID to check
            
        Returns:
            True if the token is a special token, False otherwise
        """
        return token_id in self._reverse_special_tokens
    
    def _register_special_token(self, name: str, token_id: int) -> None:
        """
        Register a special token.
        
        Args:
            name: Name of the special token
            token_id: Token ID
        """
        self._special_tokens[name] = token_id
        self._reverse_special_tokens[token_id] = name
    
    def _validate_text_input(self, text: Any) -> None:
        """Validate that input is a string."""
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, got {type(text).__name__}")
    
    def _validate_ids_input(self, ids: Any) -> None:
        """Validate that input is a list of integers."""
        if not isinstance(ids, list):
            raise TypeError(f"Expected list input, got {type(ids).__name__}")
        if not all(isinstance(id_, int) for id_ in ids):
            raise TypeError("All token IDs must be integers")
    
    def __len__(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size
    
    def __repr__(self) -> str:
        """String representation of the tokenizer."""
        return f"{self.__class__.__name__}(vocab_size={self._vocab_size}, model='{self._model_name}')"

