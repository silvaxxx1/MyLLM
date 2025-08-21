# myllm/Tokenizers/wrapper.py

import torch

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Ensure pad_token
        if not hasattr(self, "pad_token") or self.pad_token is None:
            self.pad_token = getattr(tokenizer, "pad_token", None)
            if self.pad_token is None:
                # fallback: use eos_token or <pad>
                self.pad_token = getattr(tokenizer, "eos_token", "<pad>")

        # Ensure pad_token_id
        if not hasattr(self, "pad_token_id") or self.pad_token_id is None:
            self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if self.pad_token_id is None:
                # fallback: try eos_token_id or 0
                self.pad_token_id = getattr(tokenizer, "eos_token_id", 0)

    def encode(self, text, return_tensors=None):
        ids = self.tokenizer.encode(text)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        # Convert tensor to list if necessary
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        except TypeError:
            # fallback for tokenizers that don't support skip_special_tokens
            text = self.tokenizer.decode(token_ids)
            if skip_special_tokens:
                if hasattr(self.tokenizer, "eos_token"):
                    text = text.replace(self.tokenizer.eos_token, "")
                if hasattr(self.tokenizer, "pad_token"):
                    text = text.replace(self.tokenizer.pad_token, "")
            return text
