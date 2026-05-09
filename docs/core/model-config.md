# ModelConfig

**File:** `myllm/Configs/ModelConfig.py`
**Import:** `from myllm import ModelConfig`

Central configuration dataclass for all supported transformer architectures.
Stores both architectural hyperparameters (layers, heads, dimensions) and training
hyperparameters (learning rate, weight decay) in one place.

---

## Quick usage

```python
from myllm import ModelConfig

# From registry
cfg = ModelConfig.from_name('gpt2-small')

# Custom
cfg = ModelConfig(n_layer=6, n_head=8, n_embd=512, vocab_size=50257, block_size=1024)

# Inspect
print(cfg.n_layer, cfg.n_embd, cfg.head_size)

# Memory estimate
import torch
mem = cfg.estimate_memory(dtype=torch.float16)
print(f'{mem["n_parameters"]/1e6:.1f}M params, {mem["parameters_gb"]:.2f} GB')

# Save / load
cfg.save('my_config.json')
cfg2 = ModelConfig.load('my_config.json')
```

---

## Fields

### Core architecture

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `""` | Human-readable identifier |
| `block_size` | `int` | `1024` | Maximum sequence length |
| `vocab_size` | `int` | `50257` | Token vocabulary size (GPT-2 default) |
| `padded_vocab_size` | `Optional[int]` | `None` | Padded vocab for alignment; auto-set to `vocab_size` |
| `n_layer` | `int` | `12` | Number of transformer blocks |
| `n_head` | `int` | `12` | Number of attention heads |
| `n_embd` | `int` | `768` | Hidden / embedding dimension |
| `eps` | `float` | `1e-5` | General numerical epsilon |
| `head_size` | `Optional[int]` | `None` | Per-head dim; auto-set to `n_embd // n_head` |

### Normalisation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `norm_class_name` | `"LayerNorm" \| "RMSNorm"` | `"LayerNorm"` | Norm layer type |
| `norm_eps` | `float` | `1e-5` | Epsilon for norm layers |

### Activation & MLP

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `activation` | `str` | `"gelu"` | Activation: `"gelu"`, `"silu"`, `"relu"`, etc. |
| `gelu_approx` | `str` | `"none"` | `"none"` = exact GELU, `"tanh"` = fast approx |
| `mlp_class_name` | `str` | `"GptNeoxMLP"` | MLP variant (see table below) |
| `mlp_ratio` | `float` | `4.0` | MLP hidden = `n_embd × mlp_ratio` |
| `mlp_hidden_size` | `Optional[int]` | `None` | Override MLP hidden; auto-computed if `None` |
| `intermediate_size` | `Optional[int]` | `None` | GPT-NeoX style; mirrors `mlp_hidden_size` |
| `post_mlp_norm` | `bool` | `False` | Norm after MLP output (Gemma-style) |

**MLP class options:**

| `mlp_class_name` | Formula | Architecture |
|-----------------|---------|-------------|
| `GptMLP` | `fc → GELU → proj` | GPT-2 |
| `GptNeoxMLP` | `fc → GELU → proj` (with `intermediate_size`) | GPT-NeoX |
| `LLaMAMLP` | `gate × silu(up) → down` (SwiGLU) | LLaMA 1/2/3 |
| `GemmaMLP` | LLaMA-style with GeLU gate | Gemma |
| `LLaMAMoE` | Sparse mixture-of-experts wrapper | Experimental |

### Attention

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `causal_attention` | `bool` | `True` | Autoregressive causal mask |
| `attention_bias` | `bool` | `False` | Bias in QKV/output projections (True for GPT-2) |
| `bias` | `bool` | `False` | Global bias for all linear layers |
| `n_query_groups` | `Optional[int]` | `None` | GQA groups; `None` → set to `n_head` (standard MHA) |
| `norm_qk` | `bool` | `False` | Normalize Q and K before attention |
| `post_attention_norm` | `bool` | `False` | Norm after attention output |
| `attention_scores_scalar` | `Optional[int]` | `None` | Override `1/√head_size` scale |
| `softcapping_threshold` | `Optional[float]` | `None` | Attention logit soft-cap |
| `attention_logit_softcapping` | `Optional[float]` | `None` | Per-layer logit soft-cap |

### Positional embeddings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `position_embedding` | `str` | `"learned"` | `"learned"`, `"rope"`, or `"none"` |
| `use_rope` | `bool` | `False` | Shortcut flag for RoPE |
| `rope_base` | `int` | `10000` | RoPE base frequency |
| `rotary_percentage` | `float` | `0.0` | Fraction of dims with RoPE (LLaMA uses `1.0`) |
| `learnable_pos_emb` | `bool` | `True` | Learnable position embedding table |

### Residual connections & weight sharing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `parallel_residual` | `bool` | `False` | Parallel attention+MLP paths (LLaMA style) |
| `shared_attention_norm` | `bool` | `False` | Share pre-norm between attention and MLP |
| `weight_tying` | `bool` | `True` | Tie `wte` ↔ `lm_head` weights (GPT-2 default) |
| `scale_embeddings` | `bool` | `False` | Scale `wte` output by `√n_embd` |
| `lm_head_bias` | `bool` | `False` | Bias in language model head |

### Training hyperparameters

Stored in `ModelConfig` for convenience; overridden by `TrainerConfig` during training.

| Field | Type | Default |
|-------|------|---------|
| `dropout` | `float` | `0.1` |
| `learning_rate` | `float` | `3e-4` |
| `weight_decay` | `float` | `0.1` |
| `beta1` | `float` | `0.9` |
| `beta2` | `float` | `0.999` |

### Loading flags

Passed through to `LLM` / `ModelLoader`.

| Field | Type | Default |
|-------|------|---------|
| `load_in_8bit` | `bool` | `False` |
| `load_in_4bit` | `bool` | `False` |
| `torch_dtype` | `Optional[str]` | `None` |
| `low_cpu_mem_usage` | `bool` | `True` |
| `device_map` | `Optional[str]` | `None` |

---

## Methods

### `ModelConfig.from_name(name) → ModelConfig`

```python
cfg = ModelConfig.from_name('gpt2-small')
```

Looks up `name_to_config` registry. Raises `ValueError` for unknown names.

### `ModelConfig.available_configs() → list[str]`

```python
print(ModelConfig.available_configs())
# ['gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
#  'llama2-7b', 'llama2-13b', 'llama3-1b', 'llama3-3b', 'llama3-8b']
```

### `config.estimate_memory(batch_size=1, dtype=torch.float32) → dict`

Returns memory estimates in GB:
```python
{
    "parameters_gb": 0.57,
    "activations_gb": 0.14,
    "total_gb": 0.71,
    "n_parameters": 151_936_512,
}
```

Parameter count formula:
```
n_params = n_layer × (
    4 × n_embd²          # QKV + output projection
  + 2 × n_embd × mlp_hidden_size
  + mlp_hidden_size × n_embd
) + vocab_size × n_embd  # embedding
  + 4 × n_layer × n_embd # layer norms
```

Activation estimate: `batch × seq × n_embd × n_layer × 4 × bytes_per_param`.

### `config.save(file_path)` / `ModelConfig.load(file_path)`

JSON serialise/deserialise the full config dict.

```python
cfg.save('config.json')
cfg2 = ModelConfig.load('config.json')
assert cfg2.n_layer == cfg.n_layer
```

### `config.update(**kwargs)`

In-place field update. Warns on unknown keys (does not raise).

```python
cfg.update(block_size=2048, dropout=0.0)
```

### `config.validate()`

Asserts:
- `n_embd % n_head == 0`
- `block_size > 0`
- `mlp_ratio > 0`

Called automatically in `__post_init__`.

### `config.get_trainable_params() → dict`

Returns all `int`, `float`, `bool` fields. Used for WandB `config` logging.

### Properties

```python
config.mlp_class   # → actual class (e.g. GptMLP) from myllm.model
config.norm_class  # → nn.LayerNorm or nn.RMSNorm
```

---

## Registered configs

| Name | Layers | Heads | d_model | Notes |
|------|--------|-------|---------|-------|
| `gpt2-small` | 12 | 12 | 768 | 124M; LayerNorm, learned pos, weight tying |
| `gpt2-medium` | 24 | 16 | 1024 | 335M |
| `gpt2-large` | 36 | 20 | 1280 | 774M |
| `gpt2-xl` | 48 | 25 | 1600 | 1.5B |
| `llama2-7b` | 32 | 32 | 4096 | RMSNorm, RoPE 100%, SwiGLU, no weight tying |
| `llama2-13b` | 40 | 40 | 5120 | |
| `llama3-1b` | 24 | 16 | 2048 | 128k vocab |
| `llama3-3b` | 32 | 32 | 3072 | |
| `llama3-8b` | 32 | 32 | 4096 | |

---

## Adding a new config

Add a `dict` entry to `name_to_config` in `ModelConfig.py`:

```python
dict(
    name="my-model-3b",
    block_size=4096,
    vocab_size=32000,
    n_layer=26, n_head=16, n_embd=2560,
    norm_class_name="RMSNorm",
    mlp_class_name="LLaMAMLP",
    use_rope=True,
    position_embedding="rope",
    rotary_percentage=1.0,
    weight_tying=False,
    norm_eps=1e-5,
)
```
