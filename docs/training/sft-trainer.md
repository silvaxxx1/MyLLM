# SFTTrainer

**File:** `myllm/Train/sft_trainer.py`
**Import:** `from myllm.train import SFTTrainer`

Supervised fine-tuning trainer for instruction-following tasks.
Computes loss only on response tokens, not on the instruction prefix.

---

## Quick usage

```python
from torch.utils.data import DataLoader
from myllm import ModelConfig
from myllm.train import SFTTrainer, SFTTrainerConfig

cfg = SFTTrainerConfig(
    output_dir='./output_sft',
    num_epochs=3,
    batch_size=4,
    report_to=[],                    # disable WandB
    model_config_name='gpt2-small',
)

trainer = SFTTrainer(cfg, model_config=ModelConfig.from_name('gpt2-small'))
trainer.setup_model()
trainer.setup_data(train_dataloader=my_dl)
trainer.setup_optimizer()
trainer.train()
```

---

## Constructor

```python
SFTTrainer(
    config: SFTTrainerConfig,
    model_config: ModelConfig = None,
    model: nn.Module = None,
)
```

Reads `instruction_template` and `response_template` from config.
Creates a `TrainingFlow` helper for epoch management.

---

## `setup_model`

```python
trainer.setup_model() -> nn.Module
```

Two code paths:

**External model provided:**
```python
trainer = SFTTrainer(cfg, model=my_pretrained_model)
trainer.setup_model()   # just moves to device + sets tokenizer
```

**Model created internally:**
```python
trainer = SFTTrainer(cfg, model_config=cfg_model)
trainer.setup_model()
# 1. LLM(config) — builds GPT structure
# 2. if config.pretrained_variant: llm.load(variant) — load pretrained weights
# 3. setup_tokenizer()
# 4. model.to(device)
# 5. optional: torch.compile()
```

Loading pretrained weights before SFT is the standard workflow for instruction tuning.
Set `config.pretrained_variant = 'gpt2'` to start from GPT-2 weights.

---

## Response masking

The key SFT-specific logic. Loss is computed only on response tokens.

### `_get_labels(batch)`

Returns `batch["labels"]` if present. Otherwise computes response masking:

```python
labels = self._create_response_mask(
    batch['input_ids'],
    batch.get('attention_mask'),
    batch.get('instruction', ''),
)
```

### `_create_response_mask(input_ids, attention_mask, instruction_text)`

For each sequence in the batch:
1. Decode `input_ids[i]` back to text
2. Find position of `response_marker` (e.g. `"### Response:"`) in the text
3. Re-encode everything up to and including the marker
4. Set `labels[i, :len(prefix_tokens)] = -100` (mask instruction)
5. Apply `labels[attention_mask == 0] = -100` (mask padding)

```
Input:  "### Instruction:\nWhat is 2+2?\n\n### Response:\n4"
Labels: [-100, -100, ..., -100, token_id("4")]
                                 ↑ only response is trained
```

The marker string is configurable:
```python
SFTTrainerConfig(response_template='### Response:')
```

---

## `train`

```python
trainer.train()
```

Full training loop using `TrainingFlow`:

```
setup_wandb()
for epoch in range(num_epochs):
    create_progress_bar(rich)
    training_flow.run_epoch(epoch, progress, task)
        → for each batch: training_step(batch)
        → log at logging_steps
        → evaluate at eval_steps
    training_flow.handle_end_of_epoch(epoch)
        → evaluate
        → save_checkpoint if new best
create_training_summary_table (rich)
wandb.finish()
```

---

## Dataset format

Your `DataLoader` should yield batches with:

```python
{
    "input_ids":      LongTensor(B, seq_len),   # tokenized instruction + response
    "attention_mask": LongTensor(B, seq_len),   # 1 for real tokens, 0 for padding
    "labels":         LongTensor(B, seq_len),   # (optional) pre-computed labels
}
```

If `"labels"` is absent, `_create_response_mask()` computes it automatically.

### Recommended instruction format

```
### Instruction:
{instruction text}

### Response:
{response text}
```

Change via `SFTTrainerConfig.instruction_template`.

---

## WandB metrics logged

| Metric | When |
|--------|------|
| `train/loss` | Every `logging_steps` |
| `train/lr` | Every `logging_steps` |
| `eval/loss` | Every `eval_steps` |
| `eval/perplexity` | Every `eval_steps` |

---

## Generating after training

```python
from myllm import LLM, GenerationConfig

llm = LLM(config=model_cfg, device=str(trainer.device))
llm.model = trainer.model
llm.model.eval()

result = llm.generate_text(
    '### Instruction:\nWhat is 2+2?\n\n### Response:\n',
    llm.tokenizer,
    GenerationConfig(max_length=20, temperature=0.7, top_k=50),
    skip_prompt=True,
)
print(result['text'])
```
