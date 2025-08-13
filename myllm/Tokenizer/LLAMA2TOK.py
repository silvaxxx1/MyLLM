
# ============================================================================
# myllm/tokenizers/llama2_tokenizer.py
"""
LLaMA2 tokenizer implementation using SentencePiece.
"""

from typing import List, Optional
import logging
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError:
    raise ImportError(
        "sentencepiece is required for LLaMA2Tokenizer. Install with: pip install sentencepiece"
    )

from .base import BaseTokenizer

logger = logging.getLogger(__name__)


class LLaMA2Tokenizer(BaseTokenizer):
    """
    LLaMA2 tokenizer using SentencePiece, the native tokenization method.
    
    This implementation uses the official SentencePiece model files that
    come with LLaMA and LLaMA2 releases, ensuring full compatibility.
    """
    
    def __init__(self, model_path: str, legacy: bool = False, **kwargs):
        """
        Initialize LLaMA2 tokenizer with SentencePiece model.
        
        Args:
            model_path: Path to the .model SentencePiece file
            legacy: Whether to use legacy LLaMA (vs LLaMA2) token handling
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        self.model_path = Path(model_path)
        self.legacy = legacy
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"SentencePiece model file not found: {model_path}")
        
        self._load_sentencepiece_model()
        self._setup_special_tokens()
        
        logger.info(f"Initialized LLaMA2Tokenizer from {model_path} (legacy={legacy})")
    
    def _load_sentencepiece_model(self) -> None:
        """Load the SentencePiece model."""
        try:
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(str(self.model_path))
            self._vocab_size = self.sp_model.vocab_size()
            
            # Log model info
            logger.debug(f"Loaded SentencePiece model with vocab_size={self._vocab_size}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SentencePiece model: {e}")
    
    def _setup_special_tokens(self) -> None:
        """Setup special tokens for LLaMA2."""
        # Core SentencePiece special tokens
        if self.sp_model.bos_id() != -1:
            self._register_special_token('bos', self.sp_model.bos_id())
        if self.sp_model.eos_id() != -1:
            self._register_special_token('eos', self.sp_model.eos_id())
        if self.sp_model.unk_id() != -1:
            self._register_special_token('unk', self.sp_model.unk_id())
        if self.sp_model.pad_id() != -1:
            self._register_special_token('pad', self.sp_model.pad_id())
        
        # LLaMA2-specific tokens
        # Try to find common LLaMA special tokens
        special_token_candidates = [
            ('<s>', 'bos_alt'),      # Alternative BOS representation
            ('</s>', 'eos_alt'),     # Alternative EOS representation
            ('<unk>', 'unk_alt'),    # Alternative UNK representation
            ('▁', 'space'),          # SentencePiece space character
        ]
        
        for token_str, token_name in special_token_candidates:
            try:
                # Try to encode the token string
                token_ids = self.sp_model.encode(token_str, out_type=int)
                if len(token_ids) == 1:
                    # Single token - likely a real special token
                    token_id = token_ids[0]
                    if token_name not in self._special_tokens:
                        self._register_special_token(token_name, token_id)
                        self._register_special_token(token_str, token_id)
            except:
                continue
        
        # For legacy LLaMA (original), handle differences if needed
        if self.legacy:
            logger.debug("Using legacy LLaMA token handling")
            # Original LLaMA may have different token handling
            # Add any legacy-specific token mappings here
    
    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encode text using SentencePiece.
        
        Args:
            text: Input text to tokenize
            bos: Whether to add beginning-of-sequence token
            eos: Whether to add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        self._validate_text_input(text)
        
        # Encode using SentencePiece
        try:
            token_ids = self.sp_model.encode(text, out_type=int)
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
        Decode token IDs using SentencePiece.
        
        Args:
            ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        self._validate_ids_input(ids)
        
        if not ids:
            return ""
        
        try:
            return self.sp_model.decode(ids)
        except Exception as e:
            logger.error(f"Failed to decode token IDs {ids}: {e}")
            raise ValueError(f"Unable to decode token IDs: {e}")
    
    def encode_as_pieces(self, text: str) -> List[str]:
        """
        Encode text as subword pieces (for debugging/analysis).
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of subword piece strings
        """
        self._validate_text_input(text)
        return self.sp_model.encode(text, out_type=str)
    
    def decode_pieces(self, pieces: List[str]) -> str:
        """
        Decode subword pieces back to text.
        
        Args:
            pieces: List of subword pieces
            
        Returns:
            Decoded text string
        """
        return self.sp_model.decode(pieces)
    
    def get_piece_size(self) -> int:
        """Get the number of pieces in the vocabulary."""
        return self.sp_model.get_piece_size()
    
    def id_to_piece(self, token_id: int) -> str:
        """Convert token ID to piece string."""
        return self.sp_model.id_to_piece(token_id)
    
    def piece_to_id(self, piece: str) -> int:
        """Convert piece string to token ID."""
        return self.sp_model.piece_to_id(piece)
