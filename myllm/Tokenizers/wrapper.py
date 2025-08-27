# myllm/Tokenizers/wrapper.py
import torch

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Ensure pad_token
        self.pad_token = getattr(tokenizer, "pad_token", None)
        if self.pad_token is None:
            self.pad_token = getattr(tokenizer, "eos_token", "<pad>")

        # Ensure pad_token_id
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(tokenizer, "eos_token_id", 0)

    # ------------------------
    # Encoding / Decoding
    # ------------------------
    def encode(self, text, return_tensors=None):
        if isinstance(text, list):
            ids = [self.tokenizer.encode(t) for t in text]
        else:
            ids = self.tokenizer.encode(text)

        if return_tensors == "pt":
            return torch.tensor(ids if isinstance(text, list) else [ids], dtype=torch.long)
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        except TypeError:
            text = self.tokenizer.decode(token_ids)
            if skip_special_tokens:
                if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token:
                    text = text.replace(self.tokenizer.eos_token, "")
                if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token:
                    text = text.replace(self.tokenizer.pad_token, "")
            return text

    # ------------------------
    # Batch encoding with padding and attention masks
    # ------------------------
    def batch_encode(self, texts, padding=True, return_tensors="pt"):
        """
        Encode a list of sentences into token IDs with optional padding.
        Returns a dict: {"input_ids": Tensor, "attention_mask": Tensor}
        """
        # Encode each sentence individually
        encoded = [self.encode(t) for t in texts]
        max_len = max(len(seq) for seq in encoded) if padding else None

        input_ids = []
        attention_mask = []

        for seq in encoded:
            if padding:
                pad_len = max_len - len(seq)
                input_ids.append(seq + [self.pad_token_id] * pad_len)
                attention_mask.append([1] * len(seq) + [0] * pad_len)
            else:
                input_ids.append(seq)
                attention_mask.append([1] * len(seq))

        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    # ------------------------
    # Useful Properties
    # ------------------------
    @property
    def vocab_size(self):
        return getattr(self.tokenizer, "vocab_size", None)

    @property
    def model_name(self):
        return getattr(self.tokenizer, "model_name", self.tokenizer.__class__.__name__)

    @property
    def special_tokens(self):
        return getattr(self.tokenizer, "special_tokens", None)

    # ------------------------
    # Readable Representation
    # ------------------------
    def __repr__(self):
        return f"TokenizerWrapper(model={self.model_name}, vocab_size={self.vocab_size}, pad_token={self.pad_token})"
