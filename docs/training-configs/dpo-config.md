# DPOTrainerConfig

**File:** `myllm/Train/configs/DPOConfig.py`
**Import:** `from myllm.train import DPOTrainerConfig`

Configuration for the DPO trainer. Extends `TrainerConfig` with
preference-optimisation specific fields.

---

## Fields (in addition to TrainerConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `beta` | `float` | `0.1` | KL penalty coefficient. Higher = stay closer to reference policy |
| `reference_model_path` | `Optional[str]` | `None` | Path to the frozen reference model checkpoint |
| `chosen_field` | `str` | `"chosen"` | Dataset column name for preferred responses |
| `rejected_field` | `str` | `"rejected"` | Dataset column name for rejected responses |
| `prompt_field` | `str` | `"prompt"` | Dataset column name for input prompts |

---

## Usage

```python
from myllm.train import DPOTrainerConfig

cfg = DPOTrainerConfig(
    output_dir='./dpo_output',
    num_epochs=1,
    batch_size=4,
    beta=0.1,
    reference_model_path='./sft_output/checkpoint-final',
    report_to=[],
)
```

---

## Notes

- The `reference_model_path` should point to a checkpoint of the SFT model
  that was used before DPO fine-tuning
- `beta=0.1` is the typical default from the DPO paper; lower values give
  more freedom from the reference policy
- This config is implemented; the trainer that uses it is still a stub
