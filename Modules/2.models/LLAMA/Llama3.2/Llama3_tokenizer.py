# Importing necessary libraries
import os
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Tokenizer:
    def __init__(self, model_path):
        # Ensure that the provided model path is a valid file
        assert os.path.isfile(model_path), f"Model file {model_path} not found"
        
        # Load the mergeable ranks (Byte Pair Encoding) specific to the model
        mergeable_ranks = load_tiktoken_bpe(model_path)

        # Define special tokens for LLaMA3
        # These tokens help manage text boundaries, reserved spaces, and other control features in the text generation process
        self.special_tokens = {
            "<|begin_of_text|>": 128000,  # Token to signify the beginning of the text
            "<|end_of_text|>": 128001,    # Token to signify the end of the text
            "<|start_header_id|>": 128006,  # Token for starting a header section
            "<|end_header_id|>": 128007,   # Token for ending a header section
            "<|eot_id|>": 128009,         # Token for end-of-text (e.g., indicating the end of a content block)
        }
        
        # Additional reserved tokens for model-specific functionality (e.g., controlling certain operations)
        self.special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()
        })

        # Define the tokenization pattern
        # This regular expression is more complex compared to GPT-2 and captures more varied tokenization patterns
        # Handles contractions, special characters, and complex text formats like headers or sequences
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,  # Use the name of the model file as the encoding name
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",  # Regex pattern for tokenization
            mergeable_ranks=mergeable_ranks,  # Load the BPE mergeable ranks specific to the model
            special_tokens=self.special_tokens  # Include special tokens defined earlier
        )


    def encode(self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        """
        Encodes a given text input into tokens, optionally including special tokens like bos (beginning of sequence)
        and eos (end of sequence). The `allowed_special` and `disallowed_special` parameters allow for fine control 
        over which special tokens are included.
        """
        
        tokens = []

        # Optionally add the <|begin_of_text|> token if 'bos' is True (indicating the start of the sequence)
        if bos:
            tokens = [self.special_tokens["<|begin_of_text|>"]]

        # Use the tokenizer's `encode` method to tokenize the input text
        tokens += self.model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)

        # Optionally add the <|end_of_text|> token if 'eos' is True (indicating the end of the sequence)
        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])

        return tokens

    def decode(self, tokens):
        """
        Decodes a sequence of tokens back into the original text.
        """
        return self.model.decode(tokens)
