# TrainableTokenizer

**File:** `myllm/Tokenizers/trainable_tok.py`
**Import:** `from myllm.Tokenizers.trainable_tok import TrainableTokenizer`

BPE tokenizer trained from scratch on a custom corpus.
Useful for domain-specific vocabularies (code, biology, specific languages).

---

## Usage

```python
from myllm.tokenizers import get_tokenizer

tok = get_tokenizer(
    'trainable',
    vocab_size=8000,
    min_frequency=2,
    special_tokens=['<pad>', '<unk>', '<bos>', '<eos>'],
)

# Train on a corpus
tok.train(['path/to/corpus.txt'])

# Encode / decode
ids = tok.encode('Hello world')
text = tok.decode(ids)
```

---

## Constructor parameters

| Param | Default | Description |
|-------|---------|-------------|
| `vocab_size` | `8000` | Target vocabulary size |
| `min_frequency` | `2` | Minimum pair frequency to merge |
| `special_tokens` | `['<pad>', '<unk>', '<bos>', '<eos>']` | Reserved tokens (always included) |
| `model_name` | `'trainable'` | Identifier string |

---

## Workflow

1. Collect raw text corpus
2. Instantiate `TrainableTokenizer(vocab_size=N)`
3. Call `tok.train(['file1.txt', 'file2.txt', ...])`
4. Save the trained vocabulary
5. Use for model training

---

## Notes

- Pure Python BPE — slower than tiktoken but fully transparent and modifiable
- Good for educational purposes or when you need exact control over the vocabulary
- Trained vocabulary can be serialised and reloaded
