# Training — Overview

**Directory:** `myllm/Train/`
**Import:** `from myllm.train import SFTTrainer, PretrainTrainer`

---

## Trainer hierarchy

```
BaseTrainer (ABC)  — myllm/Train/base_trainer.py
    ├── PretrainTrainer     next-token prediction
    ├── SFTTrainer          instruction fine-tuning with response masking
    ├── SFTClassifierTrainer  sequence classification
    ├── DPOTrainer          direct preference optimization  *(stub)*
    └── PPOTrainer          RLHF via PPO                   *(stub)*
```

All trainers share the same lifecycle and configuration pattern.

---

## Lifecycle

Every trainer follows this four-step setup before calling `train()`:

```python
trainer = SFTTrainer(config, model_config=cfg)

trainer.setup_model()     # create / load the GPT model
trainer.setup_data(train_dataloader=dl)
trainer.setup_optimizer() # AdamW + optional scheduler
trainer.train()           # full training loop
```

---

## What BaseTrainer provides

| Method | Description |
|--------|-------------|
| `training_step(batch)` | AMP forward + backward + grad clip + optimizer step |
| `evaluation_step(batch)` | Inference-mode forward + token-level cross-entropy |
| `evaluate()` | Full eval loop → `{"eval_loss", "perplexity"}` |
| `save_checkpoint(dir, is_best)` | Save model + training state |
| `load_checkpoint(dir)` | Restore model + training state |
| `setup_optimizer()` | AdamW + optional LinearLR warmup |
| `setup_tokenizer()` | Load tokenizer via factory + wrap |
| `setup_wandb()` | Init WandB run |
| `log_metrics(metrics, step)` | WandB + console logging |
| `update_best_metric(metrics)` | Track best eval metric |

---

## What subclasses must implement

| Method | Purpose |
|--------|---------|
| `_prepare_batch(batch)` | Move tensors to device |
| `_get_labels(batch)` | Extract or construct target labels |
| `setup_data(...)` | Attach `train_dataloader` and `eval_dataloader` |

---

## Training step internals

```python
# BaseTrainer.training_step(batch)
with autocast(device_type='cuda'):
    logits = self.model(batch['input_ids'])     # forward
    labels = self._get_labels(batch)
    logits, labels = self.align_sequence_lengths(logits, labels)
    loss = cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
    loss = loss / gradient_accumulation_steps

scaler.scale(loss).backward()

if (step + 1) % gradient_accumulation_steps == 0:
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()  # if any
```

---

## Components

| Component | File | Description |
|-----------|------|-------------|
| `TrainingFlow` | `utils/training_flow.py` | `run_epoch`, `handle_end_of_epoch` |
| `create_progress_bar` | `utils/progress_utils.py` | rich progress bars |
| `create_training_summary_table` | `utils/summary_utils.py` | rich table at end of training |
| `setup_model_compilation` | `utils/model_utils.py` | `torch.compile()` wrapper |
| `load_pretrained_weights` | `utils/model_utils.py` | Load GPT-2 weights before SFT |

---

## See also

- [BaseTrainer](base-trainer.md)
- [SFTTrainer](sft-trainer.md)
- [PretrainTrainer](pretrain-trainer.md)
- [DPOTrainer](dpo-trainer.md) *(stub)*
- [PPOTrainer](ppo-trainer.md) *(stub)*
- [TrainerConfig](../training-configs/trainer-config.md)
