# LLM

**File:** `myllm/api.py`
**Import:** `from myllm import LLM`

High-level wrapper around `GPT` for inference. Handles weight loading,
KV-cached autoregressive generation, and mixed-precision inference.

---

## Constructor

```python
LLM(
    config: ModelConfig = None,
    torch_dtype: Optional[torch.dtype] = None,
    low_cpu_mem_usage: bool = True,
    device: str = "cpu",
)
```

| Arg | Description |
|-----|-------------|
| `config` | If provided, immediately initialises the GPT model structure (no weights yet) |
| `torch_dtype` | Override dtype: `torch.float16`, `torch.bfloat16`, etc. |
| `low_cpu_mem_usage` | Stream weights layer-by-layer to save peak RAM |
| `device` | `"cpu"`, `"cuda"`, `"cuda:1"`, etc. |

Internal state after init:
- `self.model` — `GPT` instance (or `None` if no config)
- `self.config` — `ModelConfig`
- `self.tokenizer` — `None` until set by `from_pretrained()`
- `self.loader` — `ModelLoader(cache_dir="./models")`
- `self.sampler` — `OptimizedSampler()`

---

## `LLM.from_pretrained` *(classmethod)*

```python
llm = LLM.from_pretrained(
    model_name: str,
    device: Optional[str] = None,          # auto: cuda > cpu
    torch_dtype: Optional[torch.dtype] = None,
    low_cpu_mem_usage: bool = True,
) -> LLM
```

**The recommended way to create an LLM.** Does everything in one call:

1. Auto-detect device (`cuda` if available, else `cpu`)
2. `ModelConfig.from_name(model_name)`
3. `LLM(config, device, ...)`
4. `llm.load(model_name)` — download + map weights
5. `get_tokenizer(model_name.split('-')[0])` → stored as `llm.tokenizer`

```python
llm = LLM.from_pretrained('gpt2-small')
print(llm)
# LLM(model='gpt2-small', params=124.4M, device='cuda', dtype=torch.float32)
```

---

## `llm.load`

```python
llm.load(model_variant: str, model_family: Optional[str] = None)
```

Downloads weights from HuggingFace (first run only; cached to `./models/`),
maps them to the GPT architecture, and puts the model in eval mode.

`model_family` is auto-detected from `model_variant` via `MODEL_REGISTRY`.
You never need to pass it explicitly.

```python
llm = LLM(config=ModelConfig.from_name('gpt2-small'), device='cuda')
llm.load('gpt2-small')
```

**Internal loading sequence:**
```
MODEL_REGISTRY lookup → URL
→ download_safetensors (skip if cached)
→ load_safetensors to CPU
→ GPT(config).to("cpu")
→ dtype conversion (if torch_dtype set)
→ weight_mapper.map_weights(model, params, config, device)
→ del params → gc.collect() → cuda.empty_cache()
→ model.eval()
```

---

## `llm.generate`

```python
llm.generate(
    input_ids: torch.Tensor,           # (batch, seq_len)
    generation_config: GenerationConfig,
) -> dict
```

Core autoregressive generation loop. Returns:
```python
{"tokens": Tensor(batch, seq+new), "logprobs": Tensor (optional)}
```

### KV cache path (`use_kv_cache=True`)

```
initialize_kv_cache(batch, max_seq_len, dtype)
→ full prompt forward pass         # fills cache, returns logits
→ loop new_token in range(max_length):
    input = last_token only         # (batch, 1)
    pos_offset = current_len - 1
    forward(input, use_cache=True, pos_offset)
    → sample next token
    → append to generated
    → check EOS / repetition heuristic
```

**Requires** that the model's KV cache is large enough for `prompt_len + max_length`.

### No-cache path (`use_kv_cache=False`)

```
loop new_token in range(max_length):
    forward(full_generated_sequence)   # O(n²) — all tokens re-processed
    → sample from last position
```

Slower but required for batched generation (variable prompt lengths make cache indexing complex).

---

## `llm.generate_text`

```python
llm.generate_text(
    prompt: str,
    tokenizer=None,                             # uses llm.tokenizer if None
    generation_config: GenerationConfig = None, # defaults to GenerationConfig()
    skip_prompt: bool = False,                  # strip prompt from returned text
) -> dict
```

Convenience wrapper: encodes the prompt, calls `generate`, decodes the result.

```python
# Tokenizer auto-loaded by from_pretrained
result = llm.generate_text('Hello world', GenerationConfig(max_length=50))
print(result['text'])   # includes prompt

# Skip prompt — return only the new tokens
result = llm.generate_text('Hello world', GenerationConfig(max_length=50), skip_prompt=True)
print(result['text'])   # new tokens only
```

**`skip_prompt` implementation:**
```python
tokens = output["tokens"][0]
if skip_prompt:
    tokens = tokens[input_ids.shape[1]:]   # slice off prompt length
text = tok.decode(tokens, skip_special_tokens=True)
```

---

## `llm.generate_batch`

```python
llm.generate_batch(
    prompts: List[str],
    tokenizer,
    generation_config: GenerationConfig,
) -> List[dict]
```

Pads all prompts to the same length (right-pad with `pad_token_id`),
runs one batched `generate()` call, decodes each result independently.

```python
results = llm.generate_batch(
    ['Once upon a time', 'The capital of France'],
    llm.tokenizer,
    GenerationConfig(max_length=30, use_kv_cache=False),
)
for r in results:
    print(r['text'])
```

**Note:** Always use `use_kv_cache=False` for batch generation.

---

## `llm.save`

```python
llm.save(save_path: str)
```

Saves `model.state_dict()` with `torch.save`. Creates parent directories automatically.

---

## `llm.list_models`

```python
llm.list_models() -> dict[str, list[str]]
# {"gpt2": ["gpt2-small", "gpt2-medium", ...], "llama2": [...], ...}
```

Returns all model families and variants registered in `MODEL_REGISTRY`.

---

## `__repr__`

```python
print(llm)
# Before loading:  LLM(no model loaded, device='cpu')
# After loading:   LLM(model='gpt2-small', params=124.4M, device='cuda', dtype=torch.float32)
```

---

## Manual setup (without `from_pretrained`)

```python
import torch
from myllm import LLM, ModelConfig

# 1. Build empty model structure for training
llm = LLM(config=ModelConfig.from_name('gpt2-small'), device='cuda')
# model weights are random at this point

# 2. Or load pretrained weights manually
llm2 = LLM(config=ModelConfig.from_name('gpt2-medium'), device='cuda')
llm2.load('gpt2-medium')

# 3. Wrap an already-trained model for inference
from myllm.model import GPT
model = GPT(ModelConfig.from_name('gpt2-small'))
# ... train model ...
llm3 = LLM(config=ModelConfig.from_name('gpt2-small'), device='cuda')
llm3.model = model
```

---

## Thread safety

`LLM` is not thread-safe. For concurrent inference, create one instance per thread
or use a request queue with a single GPU worker.
