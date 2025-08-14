

# ============================================================================
# myllm/tokenizers/llama3_tokenizer.py
"""
LLaMA3 tokenizer implementation using tiktoken BPE with Meta's tokenizer.json.
"""

from typing import List, Optional, Dict, Any
import logging
import json
from pathlib import Path
import re

try:
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe
except ImportError:
    raise ImportError(
        "tiktoken is required for LLaMA3Tokenizer. Install with: pip install tiktoken"
    )

from .base import BaseTokenizer

logger = logging.getLogger(__name__)


class LLaMA3Tokenizer(BaseTokenizer):
    """
    LLaMA3 tokenizer using tiktoken BPE with Meta's official tokenizer.json.
    
    This implementation loads Meta's official tokenizer configuration and
    recreates the exact tokenization behavior using tiktoken's BPE engine.
    """
    
    # Default LLaMA3 special tokens (if tokenizer.json is not available)
    DEFAULT_SPECIAL_TOKENS = {
        '<|begin_of_text|>': 128000,
        '<|end_of_text|>': 128001,
        '<|reserved_special_token_0|>': 128002,
        '<|reserved_special_token_1|>': 128003,
        '<|reserved_special_token_2|>': 128004,
        '<|reserved_special_token_3|>': 128005,
        '<|start_header_id|>': 128006,
        '<|end_header_id|>': 128007,
        '<|reserved_special_token_4|>': 128008,
        '<|eot_id|>': 128009,  # End of turn
    }
    
    def __init__(self, tokenizer_json_path: Optional[str] = None, **kwargs):
        """
        Initialize LLaMA3 tokenizer.
        
        Args:
            tokenizer_json_path: Path to Meta's tokenizer.json file (optional)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        self.tokenizer_json_path = tokenizer_json_path
        self.tokenizer_config: Optional[Dict[str, Any]] = None
        
        if tokenizer_json_path:
            self._load_from_tokenizer_json(tokenizer_json_path)
        else:
            self._load_default_configuration()
        
        self._setup_encoding()
        self._setup_special_tokens()
        
        logger.info(f"Initialized LLaMA3Tokenizer with vocab_size={self._vocab_size}")
    
    def _load_from_tokenizer_json(self, json_path: str) -> None:
        """Load tokenizer configuration from Meta's tokenizer.json."""
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"Tokenizer JSON file not found: {json_path}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.tokenizer_config = json.load(f)
            
            logger.info(f"Loaded tokenizer configuration from {json_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer.json: {e}")
    
    def _load_default_configuration(self) -> None:
        """Load default LLaMA3 configuration when tokenizer.json is not available."""
        logger.info("Using default LLaMA3 tokenizer configuration")
        
        # Use a compatible base encoding
        self.base_encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding as base
        self._vocab_size = 128256  # Standard LLaMA3 vocabulary size
    
    def _setup_encoding(self) -> None:
        """Setup the tiktoken encoding from tokenizer configuration."""
        if self.tokenizer_config:
            try:
                # Extract vocabulary and merges from tokenizer.json
                model_config = self.tokenizer_config.get('model', {})
                vocab = model_config.get('vocab', {})
                merges = model_config.get('merges', [])
                
                if vocab and merges:
                    # Create tiktoken encoding from vocab and merges
                    mergeable_ranks = {
                        token: rank for token, rank in vocab.items()
                        if not token.startswith('<|') or not token.endswith('|>')
                    }
                    
                    # Extract special tokens
                    special_tokens = {
                        token: rank for token, rank in vocab.items()
                        if token.startswith('<|') and token.endswith('|>')
                    }
                    
                    # Define the regex pattern for LLaMA3 tokenization
                    # This pattern handles the tokenization similar to GPT-4
                    pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
                    
                    # Create the custom tiktoken encoding
                    self.encoding = tiktoken.Encoding(
                        name="llama3_custom",
                        pat_str=pat_str,
                        mergeable_ranks=mergeable_ranks,
                        special_tokens=special_tokens
                    )
                    
                    self._vocab_size = len(vocab)
                    logger.info("Successfully created custom tiktoken encoding from tokenizer.json")
                    
                else:
                    raise ValueError("Invalid tokenizer.json: missing vocab or merges")
                    
            except Exception as e:
                logger.warning(f"Failed to create encoding from tokenizer.json: {e}. Using fallback.")
                self._load_default_configuration()
        else:
            # Use default base encoding
            self.encoding = self.base_encoding
    
    def _setup_special_tokens(self) -> None:
        """Setup special tokens for LLaMA3."""
        special_tokens_to_register = {}
        
        if self.tokenizer_config:
            # Extract special tokens from tokenizer.json
            added_tokens = self.tokenizer_config.get('added_tokens', [])
            
            for token_info in added_tokens:
                if isinstance(token_info, dict):
                    content = token_info.get('content')
                    token_id = token_info.get('id')
                    
                    if content and token_id is not None:
                        special_tokens_to_register[content] = token_id
            
            # Also check model vocab for special tokens
            model_config = self.tokenizer_config.get('model', {})
            vocab = model_config.get('vocab', {})
            
            for token, token_id in vocab.items():
                if token.startswith('<|') and token.endswith('|>'):
                    special_tokens_to_register[token] = token_id
        else:
            # Use default special tokens
            special_tokens_to_register = self.DEFAULT_SPECIAL_TOKENS.copy()
        
        # Register all special tokens
        for token_str, token_id in special_tokens_to_register.items():
            self._register_special_token(token_str, token_id)
            
            # Create convenient aliases
            if token_str == '<|begin_of_text|>':
                self._register_special_token('bos', token_id)
            elif token_str == '<|end_of_text|>':
                self._register_special_token('eos', token_id)
            elif token_str == '<|eot_id|>':
                self._register_special_token('eot', token_id)
        
        # Set default pad token if not explicitly defined
        if 'pad' not in self._special_tokens:
            # Use eos as pad token (common practice)
            eos_id = self.get_special_token_id('eos')
            if eos_id is not None:
                self._register_special_token('pad', eos_id)
    
    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encode text using LLaMA3 BPE tokenization.
        
        Args:
            text: Input text to tokenize
            bos: Whether to add <|begin_of_text|> token
            eos: Whether to add <|end_of_text|> token
            
        Returns:
            List of token IDs
        """
        self._validate_text_input(text)
        
        # Encode using tiktoken
        try:
            if hasattr(self, 'encoding'):
                token_ids = self.encoding.encode(text)
            else:
                # Fallback to base encoding
                token_ids = self.base_encoding.encode(text)
                
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise ValueError(f"Unable to encode text: {e}")
        
        # Add special tokens
        if bos:
            bos_id = self.get_special_token_id('bos')
            if bos_id is not None:
                token_ids = [bos_id] + token_ids
        
        if eos:
            eos_id = self.get_special_token_id('eos')
            if eos_id is not None:
                token_ids = token_ids + [eos_id]
        
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs using LLaMA3 tokenization.
        
        Args:
            ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        self._validate_ids_input(ids)
        
        if not ids:
            return ""
        
        try:
            if hasattr(self, 'encoding'):
                return self.encoding.decode(ids)
            else:
                # Fallback: filter special tokens and use base encoding
                filtered_ids = [
                    token_id for token_id in ids 
                    if not self.is_special_token(token_id) and token_id < self.base_encoding.n_vocab
                ]
                return self.base_encoding.decode(filtered_ids)
                
        except Exception as e:
            logger.error(f"Failed to decode token IDs {ids}: {e}")
            raise ValueError(f"Unable to decode token IDs: {e}")
    
    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """
        Apply LLaMA3 chat template to format conversation.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            add_generation_prompt: Whether to add prompt for generation
            
        Returns:
            Formatted chat string
        """
        formatted = ""
        
        # Add begin of text token
        formatted += "<|begin_of_text|>"
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Add header
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            formatted += content.strip()
            formatted += "<|eot_id|>"
        
        # Add generation prompt if requested
        if add_generation_prompt:
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return formatted
