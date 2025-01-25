import unicodedata
from collections import defaultdict

def get_stats(ids, counts=None):
    """
    Counts the frequency of adjacent pairs of tokens.
    """
    counts = counts or defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, idx):
    """
    Merges a given pair of tokens into a single token.
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def replace_control_characters(s: str) -> str:
    """
    Replaces control characters in a string with their Unicode escape.
    """
    return "".join(
        f"\\u{ord(ch):04x}" if unicodedata.category(ch)[0] == "C" else ch
        for ch in s
    )

def render_token(t: bytes) -> str:
    """
    Converts a token into a human-readable string.
    """
    s = t.decode('utf-8', errors='replace')
    return replace_control_characters(s)

class Tokenizer:
    """Base class for tokenizers with support for byte-pair encoding."""

    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.pattern = ""  # Optional pattern for tokenization
        self.special_tokens = {}  # str -> int
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        """
        Trains the tokenizer by building a vocabulary of the specified size.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")

        # Initialize with bytes as tokens
        tokens = list(text.encode("utf-8"))
        counts = get_stats(tokens)
        while len(self.vocab) < vocab_size:
            if not counts:
                break
            # Find the most frequent pair
            pair = max(counts, key=counts.get)
            idx = len(self.vocab)
            self.merges[pair] = idx
            tokens = merge(tokens, pair, idx)
            counts = get_stats(tokens)
            self.vocab = self._build_vocab()
            if verbose:
                print(f"Added pair {pair} as token {idx}")

    def encode(self, text):
        """
        Encodes a string into a list of token IDs.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        return list(text.encode("utf-8"))

    def decode(self, ids):
        """
        Decodes a list of token IDs into a string.
        """
        if not all(isinstance(i, int) for i in ids):
            raise ValueError("All IDs must be integers.")
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def _build_vocab(self):
        """
        Constructs the vocabulary from merges and special tokens.
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves the tokenizer configuration to a model and vocab file.
        """
        model_file = file_prefix + ".model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """
        Loads a tokenizer configuration from a model file.
        """
        if not model_file.endswith(".model"):
            raise ValueError("Model file must have a .model extension.")

        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            if version != "minbpe v1":
                raise ValueError("Unsupported model version.")
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
