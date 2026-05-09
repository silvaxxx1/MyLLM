# Tokenizer Factory

**File:** `myllm/Tokenizers/factory.py`
**Import:** `from myllm.tokenizers import get_tokenizer, register_tokenizer`

Central factory for creating, caching, and registering tokenizers.

---

## `get_tokenizer`

```python
get_tokenizer(model_name: str, **kwargs) -> TokenizerWrapper
```

Returns a cached `TokenizerWrapper`. Same `(model_name, kwargs)` always returns the same object.

```python
tok  = get_tokenizer('gpt2')
tok2 = get_tokenizer('gpt2')
assert tok is tok2   # same instance

# LLaMA-2 requires a model file
tok = get_tokenizer('llama2', model_path='path/to/tokenizer.model')

# LLaMA-3 (optional custom tokenizer json)
tok = get_tokenizer('llama3')
tok = get_tokenizer('llama3', tokenizer_json_path='path/to/tokenizer.json')

# Trainable BPE
tok = get_tokenizer('trainable', vocab_size=8000)
```

**Resolution order:**
1. Check instance cache → return cached wrapper
2. Check `_TOKENIZER_REGISTRY` (custom registered tokenizers)
3. Check built-in `BUILTIN_MODELS` dict → instantiate + wrap
4. Raise `ValueError` with available model list

---

## `register_tokenizer`

```python
register_tokenizer(model_name: str, tokenizer_class: Type[BaseTokenizer])
```

Register a custom tokenizer so `get_tokenizer('my-model')` works.

```python
from myllm.tokenizers import register_tokenizer
from myllm.Tokenizers.base import BaseTokenizer

class MyBPETokenizer(BaseTokenizer):
    def encode(self, text: str, **kwargs) -> list[int]:
        ...
    def decode(self, ids: list[int]) -> str:
        ...

register_tokenizer('my-model', MyBPETokenizer)

# Now usable anywhere
tok = get_tokenizer('my-model')
```

The class must inherit from `BaseTokenizer`. Raises `TypeError` otherwise.

---

## `unregister_tokenizer`

```python
unregister_tokenizer(model_name: str) -> bool
```

Removes a registered tokenizer. Returns `True` if it existed, `False` otherwise.

---

## `list_available_models`

```python
from myllm.Tokenizers.factory import list_available_models
print(list_available_models())
# ['gpt-2', 'gpt2', 'gpt2-large', 'gpt2-medium', 'gpt2-xl',
#  'llama-2', 'llama-3', 'llama2', 'llama3', 'trainable', ...]
```

---

## `get_model_info`

```python
from myllm.Tokenizers.factory import get_model_info
info = get_model_info('gpt2')
# {"model_name": "gpt2", "vocab_size": 50257, "special_tokens": {...}}
```

---

## Caching

The cache key is `(model_key, frozenset(kwargs.items()))`. This means:
- `get_tokenizer('gpt2')` and `get_tokenizer('gpt2')` → same object
- `get_tokenizer('llama2', model_path='a.model')` and `get_tokenizer('llama2', model_path='b.model')` → different objects

The cache lives for the lifetime of the Python process. To clear it manually:
```python
from myllm.Tokenizers.factory import _INSTANCE_CACHE
_INSTANCE_CACHE.clear()
```
