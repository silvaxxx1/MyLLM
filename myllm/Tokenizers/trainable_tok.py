# ============================================================================
# myllm/tokenizers/trainable_tok.py
from .base import BaseTokenizer
from collections import Counter
from typing import List, Optional, Callable
import json
import string

class TrainableTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab_size: int = 10000,
        min_frequency: int = 1,
        max_frequency: Optional[int] = None,
        special_tokens: Optional[List[str]] = None,
        lowercase: bool = True,
        strip_punctuation: bool = False,
        max_length: Optional[int] = None,
        tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Core settings
        self._vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self.max_length = max_length

        # Tokenization
        self.tokenizer_fn = tokenizer_fn or self._default_tokenizer

        # Vocab
        self.vocab: dict[str, int] = {}
        self.reverse_vocab: dict[int, str] = {}
        self.train_counter = Counter()

        # Register default special tokens
        default_tokens = special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"]
        for i, token in enumerate(default_tokens):
            self._register_special_token(token, i)

        self.pad_token = "<pad>"
        self.token_coverage: float = 0.0

    # -------------------------
    # Tokenization Methods
    # -------------------------
    def _default_tokenizer(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        if self.strip_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    # -------------------------
    # Training
    # -------------------------
    def train(self, texts: List[str]):
        for text in texts:
            self._validate_text_input(text)
            tokens = self.tokenizer_fn(text)
            self.train_counter.update(tokens)

        # Apply frequency filtering
        tokens_filtered = [
            (t, c) for t, c in self.train_counter.items()
            if c >= self.min_frequency and (self.max_frequency is None or c <= self.max_frequency)
        ]

        # Keep most common tokens
        most_common = sorted(tokens_filtered, key=lambda x: x[1], reverse=True)[:self._vocab_size - len(self._special_tokens)]
        start_idx = len(self._special_tokens)
        for i, (token, _) in enumerate(most_common):
            self.vocab[token] = i + start_idx
            self.reverse_vocab[i + start_idx] = token

    # -------------------------
    # Encoding & Decoding
    # -------------------------
    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        self._validate_text_input(text)
        tokens = self.tokenizer_fn(text)
        ids = [self.vocab.get(t, self.get_special_token_id("<unk>")) for t in tokens]

        if bos: ids = [self.get_special_token_id("<bos>")] + ids
        if eos: ids = ids + [self.get_special_token_id("<eos>")]

        if self.max_length:
            ids = ids[:self.max_length]
            ids += [self.get_special_token_id(self.pad_token)] * (self.max_length - len(ids))

        return ids

    def decode(self, ids: List[int]) -> str:
        self._validate_ids_input(ids)
        tokens = [self.reverse_vocab.get(i, "<unk>") for i in ids if not self.is_special_token(i)]
        return " ".join(tokens)

    # -------------------------
    # Utilities
    # -------------------------
    def add_special_token(self, token_name: str) -> int:
        idx = len(self._special_tokens)
        self._register_special_token(token_name, idx)
        return idx

    def save_vocab(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
            self.reverse_vocab = {i: t for t, i in self.vocab.items()}

    def calculate_coverage(self, texts: List[str]):
        total_tokens = sum(len(self.tokenizer_fn(t)) for t in texts)
        known_tokens = sum(sum(1 for token in self.tokenizer_fn(t) if token in self.vocab) for t in texts)
        self.token_coverage = known_tokens / total_tokens if total_tokens > 0 else 0.0
        return self.token_coverage


