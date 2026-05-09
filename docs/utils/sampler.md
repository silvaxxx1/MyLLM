# OptimizedSampler

**File:** `myllm/utils/sampler.py`
**Import:** `from myllm.utils import OptimizedSampler`

Vectorised sampling utilities used by `LLM.generate()`.
All methods operate on batched logit tensors.

---

## Usage

```python
from myllm.utils import OptimizedSampler

sampler = OptimizedSampler()

# Apply top-k then top-p in one call
logits = sampler.combined_top_k_top_p_sampling(logits, top_k=50, top_p=0.95)

# Repetition penalty
logits = sampler.apply_repetition_penalty_vectorized(
    logits, generated_ids, penalty=1.2
)

# Check for EOS
done = sampler.check_eos_vectorized(next_token, eos_token_ids=[50256])
```

---

## Methods

### `combined_top_k_top_p_sampling`

```python
sampler.combined_top_k_top_p_sampling(
    logits: Tensor,          # (batch, vocab)
    top_k: Optional[int],
    top_p: Optional[float],
) -> Tensor
```

Applies top-k and top-p filtering in sequence:

1. **Top-K:** Keep only the K highest-logit tokens. Set all others to `-inf`.
2. **Top-P:** Within the remaining tokens, keep the smallest set whose cumulative
   probability ≥ p. Set all others to `-inf`.

If `top_k=None`, top-k is skipped. If `top_p=None`, top-p is skipped.
Returns filtered logits (un-softmaxed).

### `apply_repetition_penalty_vectorized`

```python
sampler.apply_repetition_penalty_vectorized(
    logits: Tensor,          # (batch, vocab)
    generated: Tensor,       # (batch, seq_len) — already generated token IDs
    penalty: float,          # > 1.0 discourages repetition
) -> Tensor
```

For each token ID that appears in `generated`, divides its logit by `penalty`
(if positive) or multiplies by `penalty` (if negative). Vectorised across
the batch using `gather` and `scatter`.

`penalty=1.0` is a no-op (identity). `penalty=1.2` gently discourages repetition.
`penalty=2.0` strongly penalises it.

### `check_eos_vectorized`

```python
sampler.check_eos_vectorized(
    next_token: Tensor,           # (batch, 1)
    eos_token_ids: list[int],
) -> bool
```

Returns `True` if **any** sequence in the batch produced an EOS token.
Used by `LLM.generate()` for early stopping when `early_stopping=True`.

---

## Design notes

- All methods are static-compatible (no stored state)
- Operate on raw logits, not probabilities — caller applies softmax after sampling
- The repetition penalty implementation is `O(batch × seq_len)` using vectorised
  scatter operations rather than a Python loop
