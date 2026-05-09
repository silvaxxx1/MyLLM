# Test Suite

**Directory:** `myllm/tests/`
**Run:** `uv run pytest` or `uv run pytest -q` for quiet output

128 tests, ~40 seconds on CPU. No GPU, no downloaded weights required.

---

## Running tests

```bash
# All tests
uv run pytest

# Single file
uv run pytest myllm/tests/test_api.py

# Stop on first failure
uv run pytest -x

# Specific test
uv run pytest myllm/tests/test_model.py::TestGPTForward::test_output_shape

# Quiet
uv run pytest -q
```

---

## Test files

### `conftest.py` — Shared fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `tiny_model_config` | session | 2-layer, 64-dim, 2-head, vocab 1000, ctx 32 |
| `tiny_gpt` | module | `GPT(tiny_model_config).eval()` |
| `greedy_gen_config` | function | `do_sample=False, use_kv_cache=False, max_length=5` |
| `pretrain_batch` | function | Tensor batch with shifted labels |
| `classification_batch` | function | Tensor batch with integer labels |
| `sft_batch` | function | Tensor batch with instruction-masked labels |

Helper functions (not fixtures):
- `make_pretrain_loader(num_samples, seq_len, vocab_size, batch_size)` — returns a `DataLoader` yielding proper tensor batches
- `make_trainer_config(config_cls, tmp_path, **overrides)` — builds a trainer config with WandB disabled and tmp output dir

### `test_config.py`

- `ModelConfig`: presets load correctly, custom init, validation (`n_embd % n_head`), update, save/load JSON, estimate_memory, get_trainable_params
- `GenerationConfig`: defaults, custom values

### `test_model.py`

- `GPT` forward pass output shape `(B, T, vocab_size)`
- All outputs finite
- Causal mask correctness
- KV cache produces identical output to full attention
- Each MLP variant (`GptMLP`, `LLaMAMLP`, etc.)
- Parameter count matches expected

### `test_api.py`

Uses `DummyTokenizer` (no tiktoken dependency):
```python
class DummyTokenizer:
    def encode(self, text, return_tensors=None): ...
    def decode(self, tokens, skip_special_tokens=True): ...
```

- `LLM` init: model not None, is `nn.Module`, device set
- `generate()`: returns dict with tokens, output longer than input, deterministic greedy, batch dimension, top-k, top-p, logprobs, token dtype
- `generate_text()`: returns dict with text, tokens when requested, empty prompt
- `generate_batch()`: returns list of dicts, handles empty prompts, single prompt

### `test_tokenizers.py`

- `GPT2Tokenizer`: encode/decode roundtrip, batch encode, special tokens
- Factory: caching (same instance returned), unknown model raises
- `TokenizerWrapper`: `batch_encode` shape, attention mask, padding

### `test_sampler.py`

- `apply_repetition_penalty_vectorized`: penalty > 1 lowers seen token logits, no-op at 1.0, negative logits
- `combined_top_k_top_p_sampling`: top-k zeroes out non-top tokens, top-p with p=1.0 is identity
- `check_eos_vectorized`: True when EOS in batch, False otherwise

### `test_training.py`

- `PretrainTrainer`: setup, single training step returns positive loss, eval loop
- `SFTTrainer`: setup with model config, training step, response masking creates `-100` labels correctly
- `SFTClassifierTrainer`: training step, classification loss

### `test_e2e.py`

Full pipeline without any mocking:
1. Model init and forward
2. Multiple training steps produce finite losses
3. Checkpoint save and reload (weights match exactly)
4. KV cache vs no-cache generate produce same shape
5. Train then generate

---

## Design principles

- **No GPU required** — all tests run on CPU with the tiny 2L/64d model
- **No downloaded weights** — random initialisation only
- **No WandB** — `report_to=[]` in all trainer configs
- **Pure tensors** — `make_pretrain_loader()` returns real tensor batches, not Python lists (avoids ToyDataset collation quirk)
- **Deterministic** — `greedy_gen_config` uses `do_sample=False` for repeatable outputs
