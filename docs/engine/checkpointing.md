# CheckpointManager

**File:** `myllm/Train/Engine/checkpoint_manager.py`

Handles saving, loading, and pruning of training checkpoints.

---

## What gets saved

Every checkpoint directory contains two files:

| File | Content |
|------|---------|
| `pytorch_model.bin` | `model.state_dict()` |
| `training_state.bin` | `{global_step, current_epoch, best_metric, optimizer, scheduler, scaler}` |

Directory naming: `{output_dir}/checkpoint-{global_step}/`

---

## BaseTrainer integration

`BaseTrainer` exposes checkpointing directly:

```python
# Save a checkpoint
path = trainer.save_checkpoint(is_best=True)
print(f'Saved to {path}')

# Resume training
trainer.load_checkpoint('./output/checkpoint-1000')
print(f'Resuming from step {trainer.global_step}')
```

---

## Resume from checkpoint

Set `resume_from_checkpoint` in the config:

```python
cfg = SFTTrainerConfig(
    output_dir='./output',
    resume_from_checkpoint='./output/checkpoint-500',
    ...
)
```

The trainer will:
1. Load model weights from `pytorch_model.bin`
2. Restore `global_step`, `current_epoch`, `best_metric`
3. Restore optimizer, scheduler, and AMP scaler state

---

## Checkpoint pruning

When `save_total_limit` is set (default `3`), old checkpoints are deleted
automatically to keep disk usage bounded. The best checkpoint (flagged with
`is_best=True`) is never deleted regardless of the limit.

---

## Manual checkpoint load

```python
import torch
from myllm import LLM, ModelConfig

llm = LLM(config=ModelConfig.from_name('gpt2-small'), device='cuda')
state = torch.load('./output/checkpoint-1000/pytorch_model.bin', map_location='cuda')
llm.model.load_state_dict(state)
llm.model.eval()
```
