# GPT2Tokenizer

**File:** `myllm/Tokenizers/gpt2_tokenizer.py`
**Backend:** `tiktoken`
**Import:** `from myllm.tokenizers import GPT2Tokenizer`

Byte-pair encoding tokenizer for GPT-2, GPT-3, and GPT-4 model families.

---

## Supported models

| `model_name` | tiktoken encoding | Vocab size |
|-------------|-------------------|------------|
| `gpt2` (default) | `gpt2` | 50,257 |
| `gpt2-medium/large/xl` | `gpt2` | 50,257 |
| `gpt3`, `davinci` | `r50k_base` | 50,257 |
| `text-davinci-003` | `p50k_base` | 50,281 |
| `gpt-3.5-turbo` | `cl100k_base` | 100,277 |
| `gpt-4`, `gpt-4-turbo` | `cl100k_base` | 100,277 |
| `gpt-4o` | `o200k_base` | 200,019 |

---

## Usage

```python
from myllm.tokenizers import GPT2Tokenizer

tok = GPT2Tokenizer()                      # default: gpt2
tok = GPT2Tokenizer(model_name='gpt-4')   # cl100k_base

ids  = tok.encode('Hello, world!')
text = tok.decode(ids)

# Add BOS/EOS
ids = tok.encode('Hello', bos=True, eos=True)

# Batch
ids_list = tok.encode_batch(['Hello', 'World'])
```

---

## Special tokens

GPT-2 uses `<|endoftext|>` (ID 50256) as BOS, EOS, PAD, and UNK.

```python
tok.get_special_token_id('eos')   # 50256
tok.get_special_token_id('bos')   # 50256
tok.get_special_token_id('pad')   # 50256
```

GPT-3.5/4 may register additional tokens: `<|im_start|>`, `<|im_end|>`, `<|im_sep|>`.

---

## `encode`

```python
tok.encode(text: str, bos: bool = False, eos: bool = False) -> list[int]
```

Calls `tiktoken.encoding.encode(text)` then optionally prepends BOS / appends EOS.

---

## `decode`

```python
tok.decode(ids: list[int]) -> str
```

Calls `tiktoken.encoding.decode(ids)`.

---

## `encode_batch`

```python
tok.encode_batch(texts: list[str], **kwargs) -> list[list[int]]
```

Calls `encode()` for each text sequentially (tiktoken has its own fast batch mode
internally).

---

## Notes

- tiktoken is significantly faster than HuggingFace's tokenizer for GPT-2 on long texts
- The encoding is identical to OpenAI's production tokenizer for the same model family
- No padding or attention mask — use `TokenizerWrapper.batch_encode()` for that
