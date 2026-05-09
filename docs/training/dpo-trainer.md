# DPOTrainer

**File:** `myllm/Train/dpo_trainer.py`
**Import:** `from myllm.train import DPOTrainer`
**Status:** Stub — scaffolding only, no real training yet.

---

## Current state

`DPOTrainer` satisfies the `BaseTrainer` ABC so the library imports cleanly,
but `train_step()` returns `{"loss": 0.0}` and no actual optimisation occurs.

```python
trainer = DPOTrainer(config, model_config=cfg)
trainer.setup_model()   # logs "DPO trainer not fully implemented yet"
trainer.train_step(batch)   # → {"loss": 0.0}
```

---

## Planned implementation

### Algorithm — Direct Preference Optimization (Rafailov et al. 2023)

DPO eliminates the need for a separate reward model by directly optimising the
policy using human preference pairs `(chosen, rejected)`.

**Loss function:**

```
L_DPO(π_θ) = -E[(x,y_w,y_l)] [
    log σ(
        β × log π_θ(y_w|x)/π_ref(y_w|x)
        - β × log π_θ(y_l|x)/π_ref(y_l|x)
    )
]
```

Where:
- `π_θ` = policy model (being trained)
- `π_ref` = reference model (frozen copy of the base model)
- `y_w` = chosen (preferred) response
- `y_l` = rejected response
- `β` = temperature controlling KL penalty (default `0.1`)

### Components to implement

1. **Reference model** — frozen copy of the base GPT loaded at init
2. **Log probability computation** — `compute_log_probs(model, input_ids, labels)`
3. **DPO loss** — using the formula above
4. **Preference dataset** — yields `{prompt, chosen, rejected}` triples
5. **Training loop** — same structure as SFTTrainer

### Dataset format (planned)

```python
{
    "input_ids_chosen":  LongTensor(B, seq_len),
    "input_ids_rejected": LongTensor(B, seq_len),
    "attention_mask_chosen":  LongTensor(B, seq_len),
    "attention_mask_rejected": LongTensor(B, seq_len),
}
```

### Config

`DPOTrainerConfig` (already implemented):

| Field | Default | Description |
|-------|---------|-------------|
| `beta` | `0.1` | KL penalty coefficient |
| `reference_model_path` | `None` | Path to frozen reference model checkpoint |
| `chosen_field` | `"chosen"` | Column name in preference dataset |
| `rejected_field` | `"rejected"` | Column name in preference dataset |
| `prompt_field` | `"prompt"` | Column name in preference dataset |

### Reference reading

- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — Rafailov et al. 2023
