# PretrainTrainer

**File:** `myllm/Train/trainer.py`
**Import:** `from myllm.train import PretrainTrainer`

Standard language model pre-training via next-token prediction.
Loss is computed on the full sequence — no masking.

---

## Quick usage

```python
from myllm import ModelConfig
from myllm.train import PretrainTrainer
from myllm.Train.configs.TrainerConfig import TrainerConfig

cfg = TrainerConfig(
    output_dir='./output_pretrain',
    num_epochs=1,
    batch_size=8,
    report_to=[],
    model_config_name='gpt2-small',
)

trainer = PretrainTrainer(cfg, model_config=ModelConfig.from_name('gpt2-small'))
trainer.setup_model()
trainer.setup_data(train_dataloader=my_dl)
trainer.setup_optimizer()
trainer.train()
```

---

## `setup_model`

Creates `GPT(model_config)`, moves to device, sets up tokenizer,
optionally compiles with `torch.compile()`.

No pretrained weight loading — model starts from random initialisation.

---

## `_get_labels(batch)`

```python
return batch.get("labels", batch["input_ids"][:, 1:].contiguous())
```

If `"labels"` is not in the batch, constructs next-token targets by shifting
`input_ids` left by one position. The final position is not trained
(it has no ground truth in an autoregressive setup).

---

## Dataset format

```python
{
    "input_ids":      LongTensor(B, seq_len),
    "attention_mask": LongTensor(B, seq_len),
    "labels":         LongTensor(B, seq_len),   # optional; auto-computed if absent
}
```

The simplest pretraining setup — tokenise raw text into fixed-length chunks,
pack them into a `DataLoader`, and pass to `trainer.setup_data()`.

---

## Difference from SFTTrainer

| | PretrainTrainer | SFTTrainer |
|--|----------------|------------|
| Loss target | Full sequence (next token) | Response tokens only |
| Pretrained weights | Never loaded | Optionally loaded before training |
| Dataset format | Raw token chunks | Instruction-response pairs |
| Typical use | Train from scratch | Fine-tune on instructions |
