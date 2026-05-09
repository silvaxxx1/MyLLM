# LLaMA2Tokenizer

**File:** `myllm/Tokenizers/llama2_tokenizer.py`
**Backend:** `sentencepiece`
**Import:** `from myllm.tokenizers import LLaMA2Tokenizer`

SentencePiece BPE tokenizer for LLaMA-2 models. Requires the official
`tokenizer.model` file from Meta (gated on HuggingFace).

---

## Usage

```python
from myllm.tokenizers import LLaMA2Tokenizer

tok = LLaMA2Tokenizer(model_path='path/to/tokenizer.model')

ids  = tok.encode('Hello, world!')
text = tok.decode(ids)
```

Via factory:
```python
from myllm.tokenizers import get_tokenizer
tok = get_tokenizer('llama2', model_path='path/to/tokenizer.model')
```

---

## Vocabulary

- Size: **32,000** tokens
- Special tokens: `<unk>` (0), `<s>` (1 = BOS), `</s>` (2 = EOS)

---

## Getting the tokenizer file

The `tokenizer.model` file is gated. You need a HuggingFace account with access to
[`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf).

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-hf tokenizer.model
```

---

## `encode`

```python
tok.encode(text: str, bos: bool = True, eos: bool = False) -> list[int]
```

LLaMA-2 conversations prepend BOS (`<s>`) by default and optionally append EOS (`</s>`).

---

## Notes

- SentencePiece handles Unicode and multilingual text robustly
- The model file is binary — not a JSON vocabulary like tiktoken
- All LLaMA-2 sizes (7B, 13B, 70B) share the same tokenizer
