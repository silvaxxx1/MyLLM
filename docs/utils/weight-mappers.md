# Weight Mappers

**File:** `myllm/utils/weight_mappers.py`
**Import:** `from myllm.utils.weight_mappers import WEIGHT_MAPPERS`

Translates HuggingFace checkpoint key names to the internal `GPT` module names,
then loads the weights layer-by-layer into the model.

---

## The problem

HuggingFace GPT-2 checkpoints use keys like:
```
transformer.h.0.attn.c_attn.weight
transformer.h.0.mlp.c_fc.weight
```

The myllm `GPT` model uses:
```
transformer.h.0.attn.qkv.weight      # (or separate q, k, v)
transformer.h.0.mlp.fc.weight
```

Each mapper handles one family's naming convention.

---

## Interface

Every mapper implements:

```python
class WeightMapper:
    def map_weights(
        self,
        model: GPT,
        params: dict[str, Tensor],
        config: ModelConfig,
        device: str,
        low_memory: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> GPT:
        ...
```

Returns the model with all weights loaded.

---

## Available mappers

| Key | Handles |
|-----|---------|
| `gpt2_mapper` | GPT-2 small/medium/large/xl |
| `llama_mapper` | LLaMA 1, 2, 3 |
| `mistral_mapper` | Mistral-7B |
| `phi_mapper` | Phi-2 |
| `gemma_mapper` | Gemma-2B, Gemma-7B |

---

## Writing a custom mapper

```python
from myllm.utils.weight_mappers import WEIGHT_MAPPERS

class MyModelMapper:
    def map_weights(self, model, params, config, device,
                    low_memory=True, torch_dtype=None):

        # Map each HF key to the corresponding myllm module
        key_map = {
            'model.embed_tokens.weight': 'transformer.wte.weight',
            'model.norm.weight': 'transformer.ln_f.weight',
            # ... layer keys
        }

        for hf_key, my_key in key_map.items():
            if hf_key in params:
                tensor = params[hf_key]
                if torch_dtype:
                    tensor = tensor.to(dtype=torch_dtype)
                # Navigate to the correct sub-module and set the parameter
                *parts, attr = my_key.split('.')
                mod = model
                for p in parts:
                    mod = getattr(mod, p)
                setattr(mod, attr, torch.nn.Parameter(tensor.to(device)))

        return model

WEIGHT_MAPPERS['my_mapper'] = MyModelMapper()
```

---

## Low memory mode

When `low_memory=True`, the mapper:
1. Iterates over layers one by one
2. Moves each layer's weights to the target device
3. Frees the CPU copy immediately

This keeps peak RAM at ≈ 2× one layer rather than 2× the full model.
