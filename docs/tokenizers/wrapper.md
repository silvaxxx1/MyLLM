# TokenizerWrapper

**File:** `myllm/Tokenizers/wrapper.py`

Provides a unified interface over any `BaseTokenizer` instance.
This is always what `get_tokenizer()` returns — you never interact with
raw tokenizer backends directly.

---

## Constructor

```python
TokenizerWrapper(tokenizer: BaseTokenizer)
```

Reads `pad_token`, `pad_token_id`, `eos_token`, `eos_token_id` from the
underlying tokenizer with sensible fallbacks.

---

## Methods

### `encode`

```python
wrapper.encode(text: str | list[str], return_tensors=None) -> list[int] | Tensor
```

```python
ids = tok.encode('Hello world')           # list[int]
ids = tok.encode('Hello world', return_tensors='pt')  # Tensor([ids])
ids = tok.encode(['Hello', 'World'])      # list of lists
```

When `return_tensors='pt'`:
- Single string → `Tensor` of shape `(1, seq_len)`
- List of strings → `Tensor` of shape `(n, seq_len)` (no padding, ragged)

### `decode`

```python
wrapper.decode(token_ids, skip_special_tokens=True) -> str
```

Accepts `list[int]` or `torch.Tensor`. Special tokens (EOS, PAD) are stripped
when `skip_special_tokens=True`. Falls back to manual string replacement if the
underlying tokenizer doesn't support the `skip_special_tokens` argument.

### `batch_encode`

```python
wrapper.batch_encode(
    texts: list[str],
    padding: bool = True,
    return_tensors: str = "pt",
    max_length: Optional[int] = None,
    truncation: bool = True,
) -> dict
```

Returns `{"input_ids": Tensor, "attention_mask": Tensor}`.

Padding is right-padding with `pad_token_id`.
Attention mask: `1` for real tokens, `0` for padding.

```python
batch = tok.batch_encode(['Hello', 'World!'], return_tensors='pt')
# {
#   "input_ids":      Tensor(2, max_len),
#   "attention_mask": Tensor(2, max_len),
# }
```

---

## Properties

| Property | Description |
|----------|-------------|
| `vocab_size` | Vocabulary size from underlying tokenizer |
| `model_name` | Model name string |
| `special_tokens` | Dict of special token name → ID |
| `pad_token_id` | Padding token ID |
| `eos_token_id` | End-of-sequence token ID |
| `eos_token` | EOS token string (e.g. `"<|endoftext|>"`) |

---

## Callable interface

`TokenizerWrapper` is callable — same as calling `.encode()`:

```python
ids = tok('Hello world')                     # list[int]
ids = tok('Hello world', return_tensors='pt')
```

---

## `__len__` and `__repr__`

```python
len(tok)    # vocab_size (fallback: 50257)
repr(tok)   # "TokenizerWrapper(model=gpt2, vocab_size=50257, pad_token=<|endoftext|>)"
```
