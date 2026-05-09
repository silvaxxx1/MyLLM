# Extension Guide

How to add new models, tokenizers, and trainers to myllm.

---

## Adding a new model

Three things are required: a `ModelConfig` entry, a `ModelRegistry` entry,
and a weight mapper.

### 1. Add the ModelConfig

In `myllm/Configs/ModelConfig.py`, add to `name_to_config`:

```python
dict(
    name="my-model-3b",
    block_size=4096,
    vocab_size=32000,
    n_layer=26,
    n_head=16,
    n_embd=2560,
    norm_class_name="RMSNorm",
    mlp_class_name="LLaMAMLP",
    use_rope=True,
    position_embedding="rope",
    rotary_percentage=1.0,
    weight_tying=False,
    norm_eps=1e-5,
),
```

Verify with:
```python
from myllm import ModelConfig
cfg = ModelConfig.from_name("my-model-3b")
print(cfg.n_layer, cfg.n_embd)
```

### 2. Add to the ModelRegistry

In `myllm/utils/model_registry.py`:

```python
MODEL_REGISTRY["my-model"] = ModelFamily(
    name="my-model",
    default_mapper="my_mapper",
    requires_auth=False,       # set True + token_env_var if gated
    variants={
        "my-model-3b": ModelSpec(
            url="https://huggingface.co/org/my-model/resolve/main/model.safetensors",
            config_name="my-model-3b",
            expected_size=6_000_000_000,
        )
    }
)
```

### 3. Write the weight mapper

In `myllm/utils/weight_mappers.py`:

```python
class MyModelMapper:
    def map_weights(self, model, params, config, device,
                    low_memory=True, torch_dtype=None):

        # Map HuggingFace key names → myllm GPT module paths
        # Inspect the HF checkpoint with: list(params.keys())
        key_map = {
            "model.embed_tokens.weight":   "transformer.wte.weight",
            "model.norm.weight":           "transformer.ln_f.weight",
            "lm_head.weight":              "lm_head.weight",
        }

        # Layer-by-layer mapping
        for layer_i in range(config.n_layer):
            hf_prefix = f"model.layers.{layer_i}"
            my_prefix  = f"transformer.h.{layer_i}"
            key_map.update({
                f"{hf_prefix}.input_layernorm.weight":  f"{my_prefix}.ln_1.weight",
                f"{hf_prefix}.self_attn.q_proj.weight": f"{my_prefix}.attn.q_proj.weight",
                # ... add all keys
            })

        for hf_key, my_key in key_map.items():
            if hf_key not in params:
                continue
            tensor = params[hf_key]
            if torch_dtype:
                tensor = tensor.to(dtype=torch_dtype)
            if low_memory:
                tensor = tensor.to(device)

            # Set parameter on the module
            *parts, attr = my_key.split(".")
            mod = model
            for p in parts:
                mod = getattr(mod, p)
            setattr(mod, attr, torch.nn.Parameter(tensor))

        return model

WEIGHT_MAPPERS["my_mapper"] = MyModelMapper()
```

**Tip:** To discover HF key names:
```python
from safetensors.torch import load_file
params = load_file("model.safetensors")
print(list(params.keys())[:20])
```

### 4. Test

```python
from myllm import LLM

llm = LLM.from_pretrained("my-model-3b")
print(llm)
result = llm.generate_text("Hello", skip_prompt=True)
print(result["text"])
```

---

## Adding a new tokenizer

### 1. Implement `BaseTokenizer`

Create `myllm/Tokenizers/my_tokenizer.py`:

```python
from typing import List
from myllm.Tokenizers.base import BaseTokenizer

class MyTokenizer(BaseTokenizer):
    def __init__(self, model_name: str = "my-model", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        # load your backend here
        self._vocab_size = 32000
        self._setup_special_tokens()

    def _setup_special_tokens(self):
        self._register_special_token("bos", 1)
        self._register_special_token("eos", 2)
        self._register_special_token("pad", 0)

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        ids = ...  # your encoding logic
        if bos:
            ids = [self.get_special_token_id("bos")] + ids
        if eos:
            ids = ids + [self.get_special_token_id("eos")]
        return ids

    def decode(self, ids: List[int]) -> str:
        return ...  # your decoding logic
```

### 2. Register in the factory

In `myllm/Tokenizers/factory.py`, add to `BUILTIN_MODELS`:

```python
BUILTIN_MODELS = {
    ...
    "my-model": MyTokenizer,
}
```

Or register at runtime:

```python
from myllm.tokenizers import register_tokenizer
register_tokenizer("my-model", MyTokenizer)
```

### 3. Export (optional)

In `myllm/Tokenizers/__init__.py`:
```python
from .my_tokenizer import MyTokenizer
```

In `myllm/__init__.py`:
```python
from .Tokenizers import MyTokenizer
```

---

## Adding a new trainer

### 1. Subclass `BaseTrainer`

```python
# myllm/Train/my_trainer.py
from myllm.Train.base_trainer import BaseTrainer
from typing import Dict, Any
import torch

class MyTrainer(BaseTrainer):

    def __init__(self, config, model_config=None, model=None):
        super().__init__(config, model_config, model)
        # trainer-specific init

    def _prepare_batch(self, batch) -> Dict[str, Any]:
        return {k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()}

    def _get_labels(self, batch) -> torch.Tensor:
        return batch["labels"]

    def setup_data(self, train_dataloader=None, eval_dataloader=None):
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

    def train(self):
        self.setup_wandb()
        for epoch in range(self.config.num_epochs):
            for batch in self.train_dataloader:
                metrics = self.training_step(batch)
                self.global_step += 1
                if self.should_log():
                    self.log_metrics({"train/loss": metrics["loss"]})
            eval_metrics = self.evaluate()
            if self.update_best_metric(eval_metrics):
                self.save_checkpoint(is_best=True)
```

### 2. Add a config (if needed)

```python
# myllm/Train/configs/MyConfig.py
from dataclasses import dataclass
from .TrainerConfig import TrainerConfig

@dataclass
class MyTrainerConfig(TrainerConfig):
    my_specific_param: float = 0.5
```

### 3. Export

In `myllm/Train/__init__.py`:
```python
from .my_trainer import MyTrainer
```

In `myllm/__init__.py`:
```python
from .Train import MyTrainer
```

### 4. Use

```python
from myllm.Train.my_trainer import MyTrainer
from myllm.Train.configs.MyConfig import MyTrainerConfig

cfg = MyTrainerConfig(output_dir='./output', num_epochs=3, report_to=[])
trainer = MyTrainer(cfg, model_config=ModelConfig.from_name('gpt2-small'))
trainer.setup_model()
trainer.setup_data(train_dataloader=dl)
trainer.setup_optimizer()
trainer.train()
```
