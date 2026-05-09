# SFTTrainerConfig

**File:** `myllm/Train/configs/SFTConfig.py`
**Import:** `from myllm.train import SFTTrainerConfig`

Extends `TrainerConfig` with fields specific to supervised fine-tuning.
Supports both instruction-following and classification tasks.

---

## Quick usage

```python
from myllm.train import SFTTrainerConfig

# Instruction following
cfg = SFTTrainerConfig(
    output_dir='./sft_output',
    num_epochs=3,
    batch_size=4,
    model_config_name='gpt2-small',
    pretrained_variant='gpt2',       # load GPT-2 weights before fine-tuning
    report_to=[],
)

# Sequence classification
cfg = SFTTrainerConfig(
    output_dir='./cls_output',
    task_type='classification',
    num_labels=2,
    label_names=['ham', 'spam'],
    model_config_name='gpt2-small',
    report_to=[],
)
```

---

## Fields (in addition to TrainerConfig)

### Task type

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task_type` | `str` | `"instruction"` | `"instruction"` or `"classification"` |
| `num_labels` | `int` | `2` | Number of classes (classification only) |

### Instruction following

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `instruction_template` | `str` | `"### Instruction:\n{instruction}\n\n### Response:\n{response}"` | Full format string for training examples |
| `response_template` | `str` | `"### Response:"` | Marker used to locate the response boundary for loss masking |
| `response_only_loss` | `bool` | `True` | Mask instruction tokens in loss (`True` = only response trained) |
| `instruction_loss_weight` | `float` | `0.0` | Weight for instruction tokens if not masking (incompatible with `response_only_loss=True`) |

### Data paths

| Field | Type | Default |
|-------|------|---------|
| `train_dataset_path` | `Optional[str]` | `None` |
| `eval_dataset_path` | `Optional[str]` | `None` |
| `max_examples` | `Optional[int]` | `None` |

### Pretrained weights

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pretrained_variant` | `Optional[str]` | `None` | Load weights before fine-tuning. e.g. `"gpt2"`, `"gpt2-medium"` |
| `pretrained_path` | `Optional[str]` | `None` | Path to local checkpoint |

### Classification

| Field | Type | Default |
|-------|------|---------|
| `classifier_dropout` | `float` | `0.1` |
| `label_names` | `Optional[List[str]]` | `None` |

### Generation (used during evaluation)

| Field | Type | Default |
|-------|------|---------|
| `max_response_length` | `int` | `100` |
| `temperature` | `float` | `0.7` |
| `top_p` | `float` | `0.9` |

---

## Validation

`SFTTrainerConfig.validate()` (called by `__post_init__`) checks:
- `task_type` is `"instruction"` or `"classification"`
- For instruction: `instruction_template` and `response_template` are non-empty
- `response_only_loss` and `instruction_loss_weight > 0` cannot both be true
- For classification: `num_labels >= 2`, `len(label_names) == num_labels` if provided
- `0 < temperature <= 2.0`, `0 < top_p <= 1.0`, `max_response_length > 0`

---

## Instruction format

Default template:
```
### Instruction:
{instruction}

### Response:
{response}
```

The `response_template` string (`"### Response:"`) is the marker that splits
instruction from response. Everything before and including this string is masked
to `-100` in the loss, so the model only learns to predict the response.

You can change both strings:
```python
SFTTrainerConfig(
    instruction_template='User: {instruction}\nAssistant: {response}',
    response_template='Assistant:',
)
```
