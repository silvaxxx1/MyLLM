# TrainerConfig

**File:** `myllm/Train/configs/TrainerConfig.py`
**Import:** `from myllm.train import TrainerConfig`

Base configuration for all trainers. Every field here is available in
`SFTTrainerConfig`, `DPOTrainerConfig`, and `PPOTrainerConfig`.

---

## Quick usage

```python
from myllm.Train.configs.TrainerConfig import TrainerConfig

cfg = TrainerConfig(
    output_dir='./my_output',
    num_epochs=3,
    batch_size=8,
    report_to=[],          # disable WandB
    model_config_name='gpt2-small',
)
```

---

## All fields

### Model

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_config_name` | `str` | `"gpt2-small"` | Name in `ModelConfig` registry |
| `model_config_path` | `Optional[str]` | `None` | Path to JSON config file (overrides name) |
| `tokenizer_name` | `str` | `"gpt2"` | Passed to `get_tokenizer()` |
| `max_seq_length` | `Optional[int]` | `None` | Truncate sequences to this length |

### Data

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_name` | `Optional[str]` | `None` | HuggingFace dataset name (for future use) |
| `data_path` | `Optional[str]` | `None` | Local dataset path |
| `preprocessing_num_workers` | `int` | `4` | DataLoader workers |

### Training loop

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_epochs` | `int` | `3` | Number of full passes over the dataset |
| `batch_size` | `int` | `8` | Per-device batch size |
| `gradient_accumulation_steps` | `int` | `1` | Effective batch = `batch_size × steps` |
| `max_grad_norm` | `float` | `1.0` | Gradient clipping max norm |
| `warmup_steps` | `int` | `0` | Linear LR warmup steps at start |
| `seed` | `int` | `42` | Random seed |

### Optimizer / Scheduler

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optimizer_type` | `OptimizerType` | `ADAMW` | `adam`, `adamw`, `sgd` |
| `scheduler_type` | `SchedulerType` | `LINEAR` | `linear`, `cosine`, `constant` |
| `learning_rate` | `Optional[float]` | `5e-5` | Overrides `ModelConfig.learning_rate` |
| `weight_decay` | `Optional[float]` | `0.01` | |
| `beta1` | `Optional[float]` | `0.9` | Adam β₁ |
| `beta2` | `Optional[float]` | `0.999` | Adam β₂ |

### Logging & checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `logging_steps` | `int` | `50` | Log metrics every N steps |
| `eval_steps` | `int` | `500` | Run evaluation every N steps |
| `save_steps` | `int` | `1000` | Save checkpoint every N steps |
| `save_total_limit` | `int` | `3` | Max checkpoints to keep |
| `output_dir` | `str` | `"./output"` | Directory for checkpoints and logs |
| `metric_for_best_model` | `str` | `"eval_loss"` | Which metric to track for best model |
| `greater_is_better` | `bool` | `False` | Direction for best metric |
| `load_best_model_at_end` | `bool` | `False` | Reload best checkpoint after training |
| `report_to` | `List[str]` | `["wandb"]` | `[]` disables all logging backends |
| `resume_from_checkpoint` | `Optional[str]` | `None` | Path to checkpoint dir to resume from |

### WandB

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wandb_project` | `Optional[str]` | `"myllm-training"` | Project name |
| `wandb_run_name` | `Optional[str]` | `None` | Run display name |
| `wandb_notes` | `Optional[str]` | `None` | Run notes |
| `wandb_tags` | `Optional[List[str]]` | `None` | Tags for filtering |
| `wandb_entity` | `Optional[str]` | `None` | Team / username |

### Hardware

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | `DeviceType \| str` | `"auto"` | `"auto"` → cuda if available |
| `mixed_precision` | `bool` | `True` | AMP autocast |
| `dataloader_num_workers` | `int` | `4` | DataLoader worker processes |
| `use_compile` | `bool` | `False` | `torch.compile()` the model |
| `use_deepspeed` | `bool` | `False` | Use DeepSpeed (requires config path) |
| `deepspeed_config_path` | `Optional[str]` | `None` | DeepSpeed JSON config |

---

## Enums

```python
from myllm.Train.configs.TrainerConfig import (
    OptimizerType,   # ADAM, ADAMW, SGD
    SchedulerType,   # LINEAR, COSINE, CONSTANT
    DeviceType,      # AUTO, CPU, CUDA
    LoggingBackend,  # WANDB, TENSORBOARD, NONE
)
```

---

## Validation (`__post_init__`)

Raises `ValueError` if:
- `batch_size <= 0`
- `num_epochs <= 0`
- `gradient_accumulation_steps <= 0`
- `use_deepspeed=True` but `deepspeed_config_path` is missing or invalid
- `"wandb" in report_to` but `wandb_project` is empty

Creates `output_dir` on disk automatically.

---

## Methods

### `to_dict() → dict`

Serialises config to a plain dict with enum values converted to strings.

### `get_wandb_config() → dict`

Returns the subset of config fields suitable for WandB `config` parameter.
