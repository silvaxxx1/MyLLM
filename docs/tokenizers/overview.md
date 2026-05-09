# Tokenizers — Overview

**Directory:** `myllm/Tokenizers/`
**Import:** `from myllm.tokenizers import get_tokenizer`

myllm provides native tokenizers for each model family, wrapped in a unified interface.
No HuggingFace tokenizer is required for core inference.

---

## Design

```
BaseTokenizer (ABC)
    ├── GPT2Tokenizer      tiktoken BPE         → gpt2, gpt-3.5, gpt-4
    ├── LLaMA2Tokenizer    SentencePiece         → llama2
    ├── LLaMA3Tokenizer    tiktoken cl100k       → llama3
    └── TrainableTokenizer custom BPE            → domain-specific

        ↓ all wrapped by ↓

TokenizerWrapper           unified .encode / .decode / .batch_encode
```

The `TokenizerWrapper` normalises differences across backends so every tokenizer
exposes the same interface regardless of the underlying library.

---

## Quick usage

```python
from myllm.tokenizers import get_tokenizer

# Always returns a TokenizerWrapper
tok = get_tokenizer('gpt2')

ids  = tok.encode('Hello, world!')
text = tok.decode(ids)
print(ids, text)

# Batch encoding with padding + attention mask
batch = tok.batch_encode(['Hello', 'World!'], return_tensors='pt')
print(batch['input_ids'].shape)     # (2, max_len)
print(batch['attention_mask'])
```

---

## Available tokenizers

| Key | Class | Backend | Vocab size |
|-----|-------|---------|------------|
| `gpt2` | `GPT2Tokenizer` | tiktoken `gpt2` | 50,257 |
| `gpt2-medium/large/xl` | `GPT2Tokenizer` | tiktoken `gpt2` | 50,257 |
| `gpt-3.5-turbo` | `GPT2Tokenizer` | tiktoken `cl100k_base` | 100,277 |
| `gpt-4` | `GPT2Tokenizer` | tiktoken `cl100k_base` | 100,277 |
| `llama2` / `llama2-7b/13b` | `LLaMA2Tokenizer` | sentencepiece | 32,000 |
| `llama3` / `llama3-8b` | `LLaMA3Tokenizer` | tiktoken | 128,256 |
| `trainable` | `TrainableTokenizer` | custom BPE | configurable |

---

## Components

| File | Purpose |
|------|---------|
| [`base.py`](../tokenizers/wrapper.md) | `BaseTokenizer` ABC |
| [`factory.py`](factory.md) | `get_tokenizer()`, `register_tokenizer()`, caching |
| [`wrapper.py`](wrapper.md) | `TokenizerWrapper` — unified interface |
| [`gpt2_tokenizer.py`](gpt2.md) | GPT-2 / GPT-4 via tiktoken |
| [`llama2_tokenizer.py`](llama2.md) | LLaMA-2 via SentencePiece |
| [`llama3_tokenizer.py`](llama3.md) | LLaMA-3 via tiktoken |
| [`trainable_tok.py`](trainable.md) | BPE trained from scratch |
