# CLAUDE.md — MyLLM Project Intelligence

## 🧠 Project Overview

**MyLLM** is a from-scratch LLM framework built for deep understanding and research.
It covers the full pipeline: tokenization → attention → training → RLHF → inference.

**Three-layer architecture:**
- `notebooks/` — guided, from-first-principles learning (21 notebooks)
- `Modules/` — isolated, focused experiments (data, models, training, finetuning, inference)
- `myllm/` — production-grade core framework (SFT, DPO, PPO, quantization, REST API)

**Primary focus:** The `myllm/` core framework — everything else serves as reference and experimentation ground for it.

---

## 📁 Project Structure

```
MyLLM/
├── notebooks/          # 21 learning notebooks (0.0 → 6.4 + Appendices)
├── Modules/
│   ├── 1.data/         # Tokenizers, dataloader, preprocessor
│   ├── 2.models/       # GPT, LLaMA2, LLaMA3.2, attention variants (MHA/MQA/GQA/Flash)
│   ├── 3.training/     # Training loops, distributed training
│   ├── 4.finetuning/   # SFT (spam, instruction), DPO, PPO, QLoRA
│   └── 5.inference/    # GPT2 inference app, quantization
├── myllm/              # Core framework
│   ├── model.py        # Core LLM definition
│   ├── api.py          # REST API (FastAPI)
│   ├── Configs/        # ModelConfig, GenConfig
│   ├── Tokenizers/     # GPT2, LLaMA2, LLaMA3, trainable tokenizer
│   ├── Train/          # SFT, DPO, PPO trainers + Engine
│   └── utils/          # Loaders, samplers, weight mappers
├── models/             # Pre-downloaded weights (GPT2 small/medium/large/xl)
└── main.py
```

---

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **Core ML:** PyTorch 2.x (pure — no HuggingFace abstractions in core)
- **Experiment tracking:** Weights & Biases (wandb)
- **Tokenizers:** tiktoken (GPT2), SentencePiece (LLaMA2), custom trainable
- **Inference UI:** Gradio
- **API:** FastAPI
- **Package manager:** uv (see `uv.lock`) — prefer `uv` over `pip`
- **Config:** YAML + Python dataclasses

---

## ⚙️ Environment Setup

```bash
# Preferred: use uv
uv sync

# Or pip
pip install -r requirements.txt

# Install as editable package
pip install -e .
```

---

## 🔑 Key Conventions

### Code Style
- Pure PyTorch — avoid HuggingFace model abstractions in `myllm/` core
- Every module should be readable standalone — no hidden magic
- Prefer explicit over implicit: name dimensions, document tensor shapes
- Use type hints throughout

### Naming
- Model configs live in `myllm/Configs/` (dataclasses)
- Trainer configs live in `myllm/Train/configs/`
- Test files are prefixed `test_` and co-located near their module
- Checkpoint dirs follow pattern: `output_{experiment_name}/checkpoint-{step}/`

### Tensor Shape Comments
Always annotate tensor shapes in model code:
```python
# x: (batch, seq_len, d_model)
x = self.attn(x)  # (batch, seq_len, d_model)
```

---

## 🚀 Common Commands

### Run a training experiment
```bash
python Modules/3.training/train.py --config configs/basic.yml
```

### Run the core framework SFT trainer
```python
from myllm import LLM, ModelConfig
from myllm.Train.sft_trainer import SFTTrainer
from myllm.Train.configs.SFTConfig import SFTTrainerConfig

model_config = ModelConfig.from_name("gpt2-small")
trainer_config = SFTTrainerConfig(output_dir="./output", report_to=[])

trainer = SFTTrainer(trainer_config, model_config=model_config)
trainer.setup_model()
trainer.setup_data(train_dataloader=my_dataloader)
trainer.train()
```

### Serve the REST API
```bash
python myllm/api.py
```

### Run all tests
```bash
uv run pytest
```

### Run specific test suites
```bash
uv run pytest myllm/tests/test_e2e.py        # end-to-end pipeline
uv run pytest myllm/tests/test_model.py      # model components
uv run pytest myllm/tests/test_training.py   # trainers
uv run pytest myllm/tests/test_api.py        # inference API
uv run pytest -x                             # stop on first failure
uv run pytest -q                             # quiet output
```

### Benchmark inference
```bash
python myllm/benchmark_api.py
```

---

## 📚 Notebook Learning Path

Follow notebooks in order for full understanding:

| Stage | Notebooks | Topic |
|-------|-----------|-------|
| 0 | `0.0.WELCOME` | Orientation |
| 1 | `1.1` → `1.2` | Data & Tokenization |
| 2 | `2.1` → `2.4` | Attention & Transformer Architectures |
| 3 | `3.1` → `3.2` | Training & Advanced Training |
| 4 | `4.1` → `4.3` | Supervised Fine-Tuning (SFT, PEFT/LoRA) |
| 5 | `5.1` → `5.2` | RLHF (PPO) & DPO |
| 6 | `6.1` → `6.4` | Inference, KV Cache, Quantization |
| A/B | Appendices | GPT-2/LLaMA2 comparison, Gradio UI |

---

## 🧩 Module Map — Key Files

| Component | Location | Notes |
|-----------|----------|-------|
| GPT model | `Modules/2.models/GPT/GPT.py` | Reference implementation |
| LLaMA 3.2 | `Modules/2.models/LLAMA/Llama3.2/Llama3.py` | Latest arch |
| Attention variants | `Modules/2.models/atten/` | MHA, MQA, GQA, Flash |
| Core model | `myllm/model.py` | Framework entry point |
| SFT Trainer | `myllm/Train/sft_trainer.py` | Main trainer |
| DPO Trainer | `myllm/Train/dpo_trainer.py` | Preference optimization |
| PPO Trainer | `myllm/Train/ppo_trainer.py` | Reinforcement learning |
| Tokenizer factory | `myllm/Tokenizers/factory.py` | Unified tokenizer interface |
| Weight loader | `myllm/utils/loader.py` | Load GPT2/LLaMA weights |
| Training Engine | `myllm/Train/Engine/trainer_engine.py` | Core training loop |
| Accelerators | `myllm/Train/Engine/accelerator/` | Single GPU, DDP, DeepSpeed, FSDP |

---

## 📦 Pre-trained Weights Available

```
models/
├── model-gpt2-small.safetensors    # 124M
├── model-gpt2-medium.safetensors   # 335M
├── model-gpt2-large.safetensors    # 774M
├── model-gpt2-xl.safetensors       # 1.5B
└── gpt2-small/model.safetensors    # Alternative format
```

Load via `myllm/utils/loader.py` or `myllm/utils/download_weight.py`.

---

## ⚠️ Important Notes

1. **No HuggingFace model wrappers in `myllm/` core** — this is intentional. Keep it pure PyTorch.
2. **`__pycache__` and `wandb/` dirs** — ignore these when reviewing code, they're artifacts.
3. **Checkpoint dirs** (`output_sft_*`, `test_enhanced_checkpoint`) — these are training outputs, not source code.
4. **WandB runs** in `Modules/4.finetuning/GPT2_RLHF_PPO/wandb/` — experiment logs only.
5. **Typo in notebooks:** `Appandix_A` and `Appandix_B` (double 'a') — don't rename, notebooks reference these paths.
6. **Duplicate model weights** exist in both `models/` (root) and `myllm/models/` — use `myllm/models/` for framework code.

---

## 🎯 Current Development Priorities

The **`myllm/` core framework is the main focus** — all work should be evaluated by how it improves the core.

1. **`myllm/model.py`** — core LLM definition, keep it clean and extensible
2. **`myllm/Train/`** — SFT, DPO, PPO trainers + Engine (accelerators, callbacks, checkpointing)
3. **`myllm/Tokenizers/`** — unified tokenizer interface across GPT2/LLaMA2/LLaMA3
4. **`myllm/api.py`** — REST API for serving
5. **`myllm/Configs/`** — centralized config management
6. **Notebooks & Modules** — stable reference material, update only when core changes require it

---

## 💡 Philosophy

> "No hidden magic. No black boxes. Every line maps to real code."

When adding features:
- Prefer clarity over cleverness
- Add shape comments to all tensor operations
- Write a test alongside every new component
- If it can be a standalone notebook experiment first — do that