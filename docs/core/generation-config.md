# GenerationConfig

**File:** `myllm/Configs/GenConfig.py`
**Import:** `from myllm import GenerationConfig`

Controls all aspects of autoregressive text generation — sampling strategy,
stopping criteria, output format, and hardware options.

---

## Quick usage

```python
from myllm import GenerationConfig

# Greedy decoding (deterministic)
cfg = GenerationConfig(max_length=50, do_sample=False)

# Sampling with top-k + top-p (recommended default)
cfg = GenerationConfig(
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    use_kv_cache=True,
)

# With repetition penalty
cfg = GenerationConfig(max_length=80, top_k=50, repetition_penalty=1.2)

# Return log-probabilities
cfg = GenerationConfig(max_length=20, return_logprobs=True)
```

---

## All fields

### Length control

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_length` | `int` | `20` | Maximum number of **new** tokens to generate |
| `max_new_tokens` | `Optional[int]` | `None` | Alias — overrides `max_length` when set |
| `min_length` | `Optional[int]` | `None` | Minimum tokens (not yet enforced) |

### Sampling

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `do_sample` | `bool` | `True` | `True` = multinomial sampling, `False` = greedy argmax |
| `temperature` | `float` | `1.0` | Divide logits by this before softmax. `< 1` = sharper, `> 1` = flatter |
| `top_k` | `Optional[int]` | `None` | Keep only the top K tokens. `None` = disabled |
| `top_p` | `Optional[float]` | `None` | Nucleus sampling: keep tokens covering cumulative prob ≥ p. `None` = disabled |
| `typical_p` | `Optional[float]` | `None` | Typical decoding *(not yet implemented)* |
| `repetition_penalty` | `float` | `1.0` | Divide logit of any previously seen token by this value. `1.0` = off, `> 1.0` = penalise |
| `no_repeat_ngram_size` | `Optional[int]` | `None` | Block repeated n-grams *(not yet implemented)* |
| `use_optimized_sampler` | `bool` | `True` | Route top-k/top-p through `OptimizedSampler` |

**Sampling is activated purely by the field value:**
- `top_k=50` → applies top-k. `top_k=None` → no top-k.
- `repetition_penalty=1.2` → active. `repetition_penalty=1.0` → no-op.

No separate `apply_*` toggle flags are needed.

### Stopping

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `early_stopping` | `bool` | `True` | Stop when any EOS token is produced |
| `eos_token_ids` | `Optional[List[int]]` | `None` | EOS token ID(s) to watch for |
| `pad_token_id` | `Optional[int]` | `None` | Pad token used in batch generation |

**Repetition heuristic:** If the last 5 generated tokens are all identical, generation
stops early regardless of `max_length`. Checked every 5 steps after the 10th token.

### Output format

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `return_tokens` | `bool` | `True` | Include token ID list in returned dict |
| `return_logprobs` | `bool` | `False` | Include log-probability tensor in returned dict |
| `output_scores` | `bool` | `False` | Reserved *(not yet implemented)* |
| `output_attentions` | `bool` | `False` | Reserved *(not yet implemented)* |
| `output_hidden_states` | `bool` | `False` | Reserved *(not yet implemented)* |
| `max_logprob_history` | `int` | `100` | Cap how many steps of log-probs are kept in memory |

### Performance

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_kv_cache` | `bool` | `True` | Pre-fill prompt into KV cache, then step one token at a time |
| `use_mixed_precision` | `bool` | `True` | FP16 autocast during generation (CUDA only) |
| `batch_size` | `int` | `1` | For batch generation (informational; not enforced by config) |

---

## Sampling order per step

At each generation step, the following operations are applied in sequence:

```
logits[:, -1, :]                    # take last token's logits
    ÷ temperature                   # scale
    → repetition_penalty            # if != 1.0
    → OptimizedSampler              # top_k → top_p (if use_optimized_sampler)
    → softmax → probabilities
    → multinomial sample            # if do_sample
    → argmax                        # if not do_sample
```

---

## Common presets

```python
# Greedy — deterministic, fast
GenerationConfig(max_length=50, do_sample=False, use_kv_cache=True)

# Creative — high diversity
GenerationConfig(max_length=200, temperature=1.2, top_p=0.95, do_sample=True)

# Focused — low temperature, tight top-k
GenerationConfig(max_length=100, temperature=0.6, top_k=20, do_sample=True)

# Balanced — recommended default
GenerationConfig(max_length=100, temperature=0.8, top_k=50, top_p=0.95, do_sample=True)

# Debug — no sampling, no cache, log probs
GenerationConfig(max_length=10, do_sample=False, use_kv_cache=False, return_logprobs=True)
```

---

## Output dictionary

`LLM.generate()` returns:

```python
{
    "tokens":   torch.Tensor,   # (batch, prompt_len + new_len)  — always present
    "logprobs": torch.Tensor,   # (batch, new_len)               — only if return_logprobs=True
}
```

`LLM.generate_text()` returns:

```python
{
    "text":     str,            # decoded text (prompt + generated, or just generated if skip_prompt=True)
    "tokens":   list | None,    # token IDs if return_tokens=True
    "logprobs": Tensor | None,  # if return_logprobs=True
}
```
