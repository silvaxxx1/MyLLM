# ModelRegistry

**File:** `myllm/utils/model_registry.py`
**Import:** `from myllm.utils.model_registry import MODEL_REGISTRY`

Defines every supported model family and variant — download URLs, config names,
weight mapper references, and authentication requirements.

---

## Data structures

```python
@dataclass
class ModelSpec:
    url: str                         # HuggingFace download URL
    config_name: str                 # Key in ModelConfig registry
    expected_size: Optional[int]     # File size check (bytes)
    weight_mapper: Optional[str]     # Override default mapper

@dataclass
class ModelFamily:
    name: str
    variants: dict[str, ModelSpec]
    default_mapper: str              # WEIGHT_MAPPERS key
    requires_auth: bool = False
    token_env_var: Optional[str] = None   # e.g. "HF_TOKEN"
```

---

## Registered families

### `gpt2` — No auth required

| Variant | Params | Config |
|---------|--------|--------|
| `gpt2-small` | 124M | `gpt2-small` |
| `gpt2-medium` | 335M | `gpt2-medium` |
| `gpt2-large` | 774M | `gpt2-large` |
| `gpt2-xl` | 1.5B | `gpt2-xl` |

Mapper: `gpt2_mapper`

### `llama2` — Requires `HF_TOKEN`

| Variant | Params | Config |
|---------|--------|--------|
| `llama2-7b` | 7B | `llama2-7b` |
| `llama2-13b` | 13B | `llama2-13b` |

Mapper: `llama_mapper`

### `llama3` — Requires `HF_TOKEN`

| Variant | Params | Config |
|---------|--------|--------|
| `llama3-1b` | 1B | `llama3-1b` |
| `llama3-8b` | 8B | `llama3-8b` |

Mapper: `llama_mapper`

### `mistral` — No auth required

| Variant | Params | Config |
|---------|--------|--------|
| `mistral-7b-v0.1` | 7B | `mistral-7b-v0.1` |

Mapper: `mistral_mapper`

### `phi` — No auth required

| Variant | Params | Config |
|---------|--------|--------|
| `phi-2` | 2.7B | `phi-2` |

Mapper: `phi_mapper`

### `gemma` — Requires `HF_TOKEN`

| Variant | Params | Config |
|---------|--------|--------|
| `gemma-2b` | 2B | `gemma-2b` |
| `gemma-7b` | 7B | `gemma-7b` |

Mapper: `gemma_mapper`

---

## Adding a new model

```python
from myllm.utils.model_registry import MODEL_REGISTRY, ModelFamily, ModelSpec

MODEL_REGISTRY["my-model"] = ModelFamily(
    name="my-model",
    default_mapper="my_mapper",       # must exist in WEIGHT_MAPPERS
    requires_auth=False,
    variants={
        "my-model-3b": ModelSpec(
            url="https://huggingface.co/org/my-model/resolve/main/model.safetensors",
            config_name="my-model-3b",  # must exist in ModelConfig registry
            expected_size=6_000_000_000,
        )
    }
)
```

Also add:
1. A `ModelConfig` entry (see [ModelConfig docs](../core/model-config.md))
2. A weight mapper (see [WeightMappers docs](weight-mappers.md))
