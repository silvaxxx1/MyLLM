# BaseTrainer

**File:** `myllm/Train/base_trainer.py`
**Import:** `from myllm.Train.base_trainer import BaseTrainer`

Abstract base class that implements the full training infrastructure.
Subclasses only need to implement three methods.

---

## Constructor

```python
BaseTrainer(
    config,            # TrainerConfig or subclass
    model_config=None, # ModelConfig (optional if passing model directly)
    model=None,        # Pre-built nn.Module (optional)
)
```

**Init sequence:**
1. `_setup_logging()` — configure Python logging, create `output_dir`
2. `_setup_device()` — resolve `DeviceType.AUTO` to `cuda`/`cpu`
3. `_setup_seed()` — `torch.manual_seed(config.seed)` + CUDA seed
4. `_setup_model_config()` — load `ModelConfig` from name/path if not passed directly;
   sync `learning_rate`, `weight_decay`, `beta1`, `beta2` from trainer config

**State attributes after init:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `self.model` | `nn.Module \| None` | Set by `setup_model()` |
| `self.optimizer` | `Optimizer \| None` | Set by `setup_optimizer()` |
| `self.scheduler` | `LRScheduler \| None` | Set by `setup_optimizer()` |
| `self.tokenizer` | `TokenizerWrapper \| None` | Set by `setup_tokenizer()` |
| `self.train_dataloader` | `DataLoader \| None` | Set by `setup_data()` |
| `self.eval_dataloader` | `DataLoader \| None` | Set by `setup_data()` |
| `self.global_step` | `int` | Incremented each optimiser update |
| `self.current_epoch` | `int` | Current epoch index |
| `self.best_metric` | `float \| None` | Best eval metric seen |
| `self.scaler` | `GradScaler` | AMP gradient scaler |
| `self.device` | `torch.device` | Target device |

---

## Abstract methods (must implement in subclass)

### `_prepare_batch(batch) → dict`

Move batch tensors to the training device.

```python
def _prepare_batch(self, batch):
    return {k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()}
```

### `_get_labels(batch) → Tensor`

Extract or compute the target labels for the current task.

- **Pretraining:** `batch["input_ids"][:, 1:]` (shifted next-token targets)
- **SFT:** `batch["labels"]` with instruction tokens masked to `-100`
- **Classification:** `batch["labels"]` (integer class indices)

### `setup_data(train_dataloader=None, eval_dataloader=None)`

Attach data loaders.

```python
def setup_data(self, train_dataloader=None, eval_dataloader=None):
    self.train_dataloader = train_dataloader
    self.eval_dataloader = eval_dataloader
```

---

## Concrete methods

### `setup_model() → nn.Module`

Default: creates `GPT(self.model_config)` and moves to device. Optionally calls
`torch.compile()` if `config.use_compile=True`. Subclasses override this to add
pretrained weight loading, classifier heads, etc.

### `setup_tokenizer()`

Calls `get_tokenizer(config.tokenizer_name)`, wraps in `TokenizerWrapper`,
and sets `pad_token` / `pad_token_id` (fallback to `eos_token_id`).

### `setup_optimizer()`

Creates `AdamW` with parameters from `model_config`:
```python
AdamW(
    model.parameters(),
    lr=model_config.learning_rate,
    weight_decay=model_config.weight_decay,
    betas=(model_config.beta1, model_config.beta2),
)
```

If `config.scheduler_type == "linear"` and `config.warmup_steps > 0`,
adds a `LinearLR` warmup scheduler.

### `training_step(batch) → dict`

```python
{"loss": float}
```

Full AMP training step with gradient accumulation. See [Overview](overview.md)
for the detailed sequence.

### `evaluation_step(batch) → dict`

```python
{"loss": float, "tokens": int}
```

Token-level cross-entropy under `torch.inference_mode()` + AMP autocast.

### `evaluate() → dict`

Iterates `eval_dataloader`, accumulates token-level loss, returns:
```python
{"eval_loss": float, "perplexity": float}
```

### `compute_loss(logits, labels) → Tensor`

```python
F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=-100,
)
```

### `align_sequence_lengths(logits, labels) → (Tensor, Tensor)`

Trims both tensors to the shorter length to handle off-by-one differences
between model output and labels.

### `save_checkpoint(checkpoint_dir=None, is_best=False) → str`

Saves two files:
- `pytorch_model.bin` → `model.state_dict()`
- `training_state.bin` → `{global_step, current_epoch, best_metric, optimizer, scheduler, scaler}`

If `checkpoint_dir` is `None`, uses `output_dir/checkpoint-{global_step}`.

### `load_checkpoint(checkpoint_dir: str)`

Loads model weights and all training state, allowing training to resume exactly.

### `setup_wandb()`

Initialises a WandB run if `"wandb" in config.report_to`. Reads:
`wandb_project`, `wandb_run_name`, `wandb_notes`, `wandb_tags` from config.

### `log_metrics(metrics: dict, step=None)`

Logs to WandB (if active) and Python logger.

### `update_best_metric(metrics: dict, key="eval_loss") → bool`

Compares current value against `self.best_metric`. Returns `True` if it's a new best.
Direction controlled by `config.greater_is_better`.

### Control predicates

```python
trainer.should_log()               # global_step % logging_steps == 0
trainer.should_evaluate()          # eval_steps > 0 and eval_dataloader present
trainer.should_save_checkpoint()   # save_steps > 0
```

---

## Multi-GPU readiness

`BaseTrainer` stores `local_rank` and `world_size` from config (defaults: 0 and 1).
When `world_size > 1`, `self.distributed = True` is set — actual DDP wrapping
is handled by the `TrainerEngine` + accelerator layer.
