
# ============================================================================
# myllm/tokenizers/gpt2_tokenizer.py
"""
GPT-2 tokenizer implementation using tiktoken BPE.
"""

from typing import List
import logging

try:
    import tiktoken
except ImportError:
    raise ImportError(
        "tiktoken is required for GPT2Tokenizer. Install with: pip install tiktoken"
    )

from .base import BaseTokenizer

logger = logging.getLogger(__name__)


class GPT2Tokenizer(BaseTokenizer):
    """
    GPT-2 family tokenizer using tiktoken's native BPE implementation.
    
    Supports GPT-2, GPT-3, GPT-3.5, and GPT-4 model families using their
    respective tiktoken encodings for maximum compatibility.
    """
    
    # Model name mappings for tiktoken
    MODEL_MAPPINGS = {
        'gpt2': 'gpt2',
        'gpt-2': 'gpt2', 
        'gpt2-medium': 'gpt2',
        'gpt2-large': 'gpt2',
        'gpt2-xl': 'gpt2',
        'gpt3': 'davinci',
        'gpt-3': 'davinci',
        'text-davinci-003': 'text-davinci-003',
        'gpt-3.5-turbo': 'gpt-3.5-turbo',
        'gpt-4': 'gpt-4',
        'gpt-4-turbo': 'gpt-4-turbo-preview',
        'gpt-4o': 'gpt-4o'
    }
    
    def __init__(self, model_name: str = 'gpt2', **kwargs):
        """
        Initialize GPT-2 tokenizer using tiktoken.
        
        Args:
            model_name: Model identifier (e.g., 'gpt2', 'gpt-4')
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name=model_name, **kwargs)
        
        self.model_name = model_name.lower()
        self._load_encoding()
        self._setup_special_tokens()
        
        logger.info(f"Initialized GPT2Tokenizer for '{model_name}' with vocab_size={self._vocab_size}")
    
    def _load_encoding(self) -> None:
        """Load the appropriate tiktoken encoding for the model."""
        tiktoken_model = self.MODEL_MAPPINGS.get(self.model_name, self.model_name)
        
        try:
            # Try to get model-specific encoding first
            if tiktoken_model in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview', 'gpt-4o']:
                self.encoding = tiktoken.encoding_for_model(tiktoken_model)
            else:
                # Fall back to base encodings
                encoding_map = {
                    'gpt2': 'gpt2',
                    'davinci': 'r50k_base',
                    'text-davinci-003': 'p50k_base'
                }
                encoding_name = encoding_map.get(tiktoken_model, 'gpt2')
                self.encoding = tiktoken.get_encoding(encoding_name)
                
        except Exception as e:
            logger.warning(f"Failed to load encoding for {tiktoken_model}: {e}. Using gpt2 encoding.")
            self.encoding = tiktoken.get_encoding('gpt2')
        
        self._vocab_size = self.encoding.n_vocab
    
    def _setup_special_tokens(self) -> None:
        """Setup special tokens for GPT-2 family models."""
        # GPT-2 uses <|endoftext|> for multiple purposes
        try:
            endoftext_tokens = self.encoding.encode("<|endoftext|>")
            if endoftext_tokens:
                endoftext_id = endoftext_tokens[0]
            else:
                # Fallback: try to find in special tokens
                endoftext_id = self.encoding.encode_ordinary("<|endoftext|>")[0] if self.encoding.encode_ordinary("<|endoftext|>") else 50256
        except:
            endoftext_id = 50256  # Default GPT-2 <|endoftext|> token ID
        
        # Register core special tokens
        self._register_special_token('bos', endoftext_id)
        self._register_special_token('eos', endoftext_id)
        self._register_special_token('pad', endoftext_id)
        self._register_special_token('unk', endoftext_id)
        self._register_special_token('endoftext', endoftext_id)
        
        # Model-specific special tokens
        if 'gpt-4' in self.model_name or 'gpt-3.5' in self.model_name:
            # These models may have additional special tokens
            special_tokens_to_check = [
                '<|im_start|>', '<|im_end|>', '<|im_sep|>'
            ]
            
            for token_str in special_tokens_to_check:
                try:
                    token_ids = self.encoding.encode(token_str)
                    if len(token_ids) == 1:
                        # Single token, register it
                        clean_name = token_str.strip('<|>').lower()
                        self._register_special_token(clean_name, token_ids[0])
                        self._register_special_token(token_str, token_ids[0])
                except:
                    continue
    
    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encode text using tiktoken BPE.
        
        Args:
            text: Input text to tokenize
            bos: Whether to add beginning-of-sequence token
            eos: Whether to add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        self._validate_text_input(text)
        
        # Encode using tiktoken
        try:
            token_ids = self.encoding.encode(text)
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
        Decode token IDs using tiktoken.
        
        Args:
            ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        self._validate_ids_input(ids)
        
        if not ids:
            return ""
        
        try:
            return self.encoding.decode(ids)
        except Exception as e:
            logger.error(f"Failed to decode token IDs {ids}: {e}")
            raise ValueError(f"Unable to decode token IDs: {e}")
    
    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        """
        Encode multiple texts efficiently.
        
        Args:
            texts: List of input texts
            **kwargs: Additional arguments passed to encode()
            
        Returns:
            List of token ID lists
        """
        return [self.encode(text, **kwargs) for text in texts]

