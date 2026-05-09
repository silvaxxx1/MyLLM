# LLaMA3Tokenizer

**File:** `myllm/Tokenizers/llama3_tokenizer.py`
**Backend:** `tiktoken`
**Import:** `from myllm.tokenizers import LLaMA3Tokenizer`

Tiktoken-based tokenizer for LLaMA-3 models. Significantly larger vocabulary
than LLaMA-2 (128k vs 32k), enabling better multilingual and code coverage.

---

## Usage

```python
from myllm.tokenizers import LLaMA3Tokenizer

tok = LLaMA3Tokenizer()

# With custom special token definitions
tok = LLaMA3Tokenizer(tokenizer_json_path='path/to/tokenizer.json')

ids  = tok.encode('Hello, world!')
text = tok.decode(ids)
```

Via factory:
```python
from myllm.tokenizers import get_tokenizer
tok = get_tokenizer('llama3')
```

---

## Vocabulary

- Size: **128,256** tokens
- Uses tiktoken's extended encoding with LLaMA-3 special tokens
- Special tokens include: `<|begin_of_text|>`, `<|end_of_text|>`, `<|eot_id|>`,
  `<|start_header_id|>`, `<|end_header_id|>`

---

## Notes

- All LLaMA-3 sizes (1B, 3B, 8B, 70B) share the same tokenizer
- The larger vocabulary reduces sequence length for multilingual and code text
  compared to LLaMA-2
- `tokenizer_json_path` is optional; if the file doesn't exist, a default
  LLaMA-3 configuration is used
