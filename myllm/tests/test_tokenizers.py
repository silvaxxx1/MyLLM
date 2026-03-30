"""Tests for the tokenizer factory and GPT-2 tokenizer."""
import pytest
from myllm.Tokenizers import get_tokenizer, list_available_models


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestTokenizerFactory:

    def test_list_available_models_returns_list(self):
        models = list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_gpt2_is_available(self):
        assert "gpt2" in list_available_models()

    def test_get_known_tokenizer(self):
        tok = get_tokenizer("gpt2")
        assert tok is not None

    def test_unknown_tokenizer_raises(self):
        with pytest.raises(Exception):
            get_tokenizer("this-tokenizer-does-not-exist-xyz")

    def test_returned_tokenizer_has_encode_decode(self):
        """Factory returns an object with encode/decode — duck-type check."""
        tok = get_tokenizer("gpt2")
        assert callable(getattr(tok, "encode", None))
        assert callable(getattr(tok, "decode", None))

    def test_caching_returns_same_instance(self):
        tok1 = get_tokenizer("gpt2")
        tok2 = get_tokenizer("gpt2")
        assert tok1 is tok2


# ---------------------------------------------------------------------------
# GPT-2 tokenizer
# ---------------------------------------------------------------------------

class TestGPT2Tokenizer:

    @pytest.fixture(scope="class")
    def tok(self):
        return get_tokenizer("gpt2")

    def test_encode_returns_list(self, tok):
        ids = tok.encode("Hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_encode_produces_ints(self, tok):
        ids = tok.encode("Hello world")
        assert all(isinstance(i, int) for i in ids)

    def test_decode_roundtrip(self, tok):
        text = "Hello world"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert text.lower() in decoded.lower()

    def test_empty_string_encodes(self, tok):
        ids = tok.encode("")
        assert isinstance(ids, list)

    def test_long_text_encodes(self, tok):
        text = "The quick brown fox jumps over the lazy dog " * 20
        ids = tok.encode(text)
        assert len(ids) > 50

    def test_vocab_size_positive(self, tok):
        assert tok.vocab_size > 0

    def test_vocab_size_reasonable(self, tok):
        # GPT-2 has 50257 tokens
        assert tok.vocab_size >= 1000

    def test_has_eos_token(self, tok):
        has_eos = hasattr(tok, "eos_token_id") or hasattr(tok, "eot_id")
        assert has_eos

    def test_special_chars_encode(self, tok):
        ids = tok.encode("Hello\nWorld\t!")
        assert len(ids) > 0

    def test_unicode_encodes(self, tok):
        ids = tok.encode("こんにちは")
        assert len(ids) > 0
