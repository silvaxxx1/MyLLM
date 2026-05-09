# ModelLoader

**File:** `myllm/utils/loader.py`
**Import:** `from myllm.utils.loader import ModelLoader`

Downloads, caches, and loads pretrained model weights into the `GPT` architecture.

---

## Constructor

```python
ModelLoader(cache_dir: str = "./models")
```

Creates `cache_dir` if it doesn't exist.

---

## `load`

```python
loader.load(
    model_variant: str,                         # e.g. "gpt2-small"
    device: str = "cuda",
    model_family: Optional[str] = None,         # auto-detected if None
    custom_config: Optional[ModelConfig] = None,
    torch_dtype: Optional[torch.dtype] = None,
    low_cpu_mem_usage: bool = True,
) -> tuple[GPT, ModelConfig]
```

### Loading sequence

```
1. _detect_family(variant)           → e.g. "gpt2"
2. MODEL_REGISTRY[family]            → ModelFamily
3. family.variants[variant]          → ModelSpec (url, config_name, mapper)
4. check requires_auth               → read HF_TOKEN if needed
5. download_safetensors(url, cache)  → filepath (skip if cached)
6. load_safetensors(filepath, "cpu") → raw param dict
7. GPT(config).to("cpu")            → model structure
8. model.to(dtype=torch_dtype)      → optional dtype cast
9. WEIGHT_MAPPERS[mapper].map_weights(model, params, config, device)
10. del params → gc.collect() → cuda.empty_cache()
11. return model, config
```

### Memory-optimised loading (`low_cpu_mem_usage=True`)

Weights are loaded to CPU first, then the weight mapper moves each tensor to the
target device one layer at a time. Peak RAM usage is roughly 2× one layer's size
rather than 2× the full model.

When `low_cpu_mem_usage=False`, all weights are loaded to the target device at once.

---

## `list_available_models`

```python
loader.list_available_models()
# {"gpt2": ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
#  "llama2": ["llama2-7b", "llama2-13b"], ...}
```

---

## Authentication

Models in `llama2`, `llama3`, and `gemma` families require a HuggingFace token:

```bash
export HF_TOKEN=hf_...
```

The loader reads this from the environment. Raises `RuntimeError` if missing.

---

## Cache directory structure

```
./models/
├── model-gpt2-small.safetensors
├── model-gpt2-medium.safetensors
├── model-llama2-7b.safetensors
└── ...
```

Files are named `model-{variant}.safetensors`. If a file already exists and
passes size verification, the download is skipped.
