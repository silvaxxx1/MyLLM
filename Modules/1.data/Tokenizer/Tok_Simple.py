from Tok_Base import Tokenizer, get_stats, merge

import sys
print(sys.path)


class BasicTokenizer(Tokenizer):
    """
    A basic tokenizer implementing Byte Pair Encoding (BPE) for text tokenization.
    This class trains the tokenizer to split text into subword units and merge the most frequent pairs.
    """

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        """
        Trains the tokenizer by building a vocabulary of the specified size
        using Byte Pair Encoding (BPE).
        
        Args:
            text (str): The input text used to train the tokenizer.
            vocab_size (int): The target vocabulary size, must be at least 256.
            verbose (bool): Whether to print detailed progress during training.
        """
        assert vocab_size >= 256, "Vocabulary size must be at least 256."

        # Initialize the training process
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")  # Convert the text to raw bytes
        ids = list(text_bytes)  # List of byte integers (range 0..255)

        # Initialize the merge and vocabulary dictionaries
        merges = {}  # (int, int) -> int (stores merge rules)
        vocab = {idx: bytes([idx]) for idx in range(256)}  # {int -> bytes}

        # Iteratively merge the most frequent pairs
        for i in range(num_merges):
            # Count the occurrences of all adjacent token pairs
            stats = get_stats(ids)

            # Check if stats is empty
            if not stats:
                print(f"Warning: No token pairs found at merge step {i+1}.")
                break  # Exit early if no pairs to merge

            # Find the pair with the highest count
            pair = max(stats, key=stats.get)

            # Assign the next available token index
            idx = 256 + i

            # Merge the pair by replacing it with the new token (idx)
            ids = merge(ids, pair, idx)

            # Save the merge rule
            merges[pair] = idx

            # Update the vocabulary with the new token
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # Print progress if verbose mode is enabled
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # Save the trained merges and vocabulary
        self.merges = merges  # Stores the merge rules (used in encode())
        self.vocab = vocab    # Stores the vocabulary mapping (used in decode())

    def decode(self, ids):
        """
        Decodes a list of token IDs into a human-readable string.
        
        Args:
            ids (list): List of token IDs (integers).
        
        Returns:
            str: The decoded text string.
        """
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        """
        Encodes a string into a list of token IDs using the trained merges.
        
        Args:
            text (str): The input text to be tokenized.
        
        Returns:
            list: A list of token IDs (integers).
        """
        text_bytes = text.encode("utf-8")  # Convert the text to raw bytes
        ids = list(text_bytes)  # List of byte integers (range 0..255)

        # Iteratively apply merges to tokenize the text
        while len(ids) >= 2:
            # Count the frequency of adjacent token pairs
            stats = get_stats(ids)

            # Check if stats is empty
            if not stats:
                break  # No pairs to merge, stop early

            # Find the pair with the lowest merge index (priority given to frequent pairs)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # If no more merges are possible, terminate the process
            if pair not in self.merges:
                break

            # Otherwise, merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids


# Sample usage of the BasicTokenizer class

# Instantiate the tokenizer
tokenizer = BasicTokenizer()

# Training data (sample text)
text = "hello world! this is a test for the basic tokenizer."

# Train the tokenizer with a desired vocabulary size
tokenizer.train(text, vocab_size=512, verbose=True)

# Encode a sentence into token IDs
encoded = tokenizer.encode("hello world!")
print("Encoded:", encoded)

# Decode the token IDs back into a string
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)
