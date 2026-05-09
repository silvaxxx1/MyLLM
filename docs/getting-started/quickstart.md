# Quickstart

## Inference in 2 lines

```python
from myllm import LLM, GenerationConfig

llm = LLM.from_pretrained('gpt2-small')
print(llm.generate_text('The future of AI is', GenerationConfig(max_length=50))['text'])
```

`from_pretrained` handles everything: config, weight download, tokenizer.

---

## Step-by-step

### 1. Load a model

```python
from myllm import LLM

llm = LLM.from_pretrained('gpt2-small')
print(llm)
# LLM(model='gpt2-small', params=124.4M, device='cuda', dtype=torch.float32)
```

Available models: `gpt2-small`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`,
`llama2-7b`, `llama2-13b`, `llama3-1b`, `llama3-3b`, `llama3-8b`.

### 2. Generate text

```python
from myllm import GenerationConfig

# String in, string out — tokenizer is auto-loaded
result = llm.generate_text(
    'Once upon a time',
    GenerationConfig(max_length=60, temperature=0.8, top_k=50, top_p=0.95),
)
print(result['text'])

# Get only the new tokens (exclude the prompt)
result = llm.generate_text(
    'Once upon a time',
    GenerationConfig(max_length=60),
    skip_prompt=True,
)
print(result['text'])
```

### 3. Batch generation

```python
results = llm.generate_batch(
    ['Hello world', 'The capital of France'],
    llm.tokenizer,
    GenerationConfig(max_length=30, use_kv_cache=False),
)
for r in results:
    print(r['text'])
```

---

## Fine-tune with SFT

```python
from torch.utils.data import DataLoader
from myllm import ModelConfig
from myllm.train import SFTTrainer, SFTTrainerConfig

trainer_cfg = SFTTrainerConfig(
    output_dir='./my_output',
    num_epochs=3,
    batch_size=4,
    report_to=[],           # disable WandB
    model_config_name='gpt2-small',
)

trainer = SFTTrainer(trainer_cfg, model_config=ModelConfig.from_name('gpt2-small'))
trainer.setup_model()
trainer.setup_data(train_dataloader=my_dataloader)
trainer.setup_optimizer()
trainer.train()
```

---

## Inspect a model config

```python
import torch
from myllm import ModelConfig

cfg = ModelConfig.from_name('gpt2-medium')
print(f'{cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd}d')

mem = cfg.estimate_memory(dtype=torch.float16)
print(f'{mem["n_parameters"]/1e6:.0f}M params, {mem["parameters_gb"]:.2f} GB (fp16)')
```

---

## Work with tokenizers directly

```python
from myllm.tokenizers import get_tokenizer

tok = get_tokenizer('gpt2')
ids = tok.encode('Hello, world!')
print(ids)
print(tok.decode(ids))

batch = tok.batch_encode(['Hello', 'World'], return_tensors='pt')
print(batch['input_ids'].shape)   # (2, max_len)
```

---

## See also

- [ModelConfig reference](../core/model-config.md)
- [GenerationConfig reference](../core/generation-config.md)
- [LLM reference](../core/llm.md)
- [SFTTrainer reference](../training/sft-trainer.md)
