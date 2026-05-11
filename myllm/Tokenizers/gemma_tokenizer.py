# ============================================================================
# myllm/tokenizers/gemma_tokenizer.py
"""
Gemma tokenizer — SentencePiece with Gemma-specific special tokens.
Inherits all encode/decode logic from LLaMA2Tokenizer; only special-token
setup differs (Gemma uses <bos>/<eos> and instruction turn tokens).
"""

from .llama2_tokenizer import LLaMA2Tokenizer
import logging

logger = logging.getLogger(__name__)


class GemmaTokenizer(LLaMA2Tokenizer):
    """
    Tokenizer for Google Gemma models.

    Gemma uses SentencePiece (vocab_size=256000) with the same encode/decode
    interface as LLaMA 2, but different special tokens:
      - Core:        <bos>=2, <eos>=1, <pad>=0, <unk>=3
      - Instruction: <start_of_turn>=106, <end_of_turn>=107

    The instruction tokens are only needed for chat/instruction-tuned variants;
    bare Gemma generation works without them.
    """

    # Gemma instruction-format token strings
    GEMMA_SPECIAL = ["<start_of_turn>", "<end_of_turn>", "<pad>", "<unk>"]

    def _setup_special_tokens(self) -> None:
        # Core tokens — read directly from the SentencePiece model
        for name, id_fn in [
            ("bos", self.sp_model.bos_id),
            ("eos", self.sp_model.eos_id),
            ("unk", self.sp_model.unk_id),
            ("pad", self.sp_model.pad_id),
        ]:
            token_id = id_fn()
            if token_id != -1:
                self._register_special_token(name, token_id)

        # Gemma-specific tokens — look up by piece string
        for token_str in self.GEMMA_SPECIAL:
            token_id = self.sp_model.piece_to_id(token_str)
            if token_id != 0:  # 0 means not found in SentencePiece
                self._register_special_token(token_str, token_id)
                logger.debug(f"Registered Gemma token '{token_str}' → {token_id}")
