# MyLLM — A Transparent LLM Framework, Built From Scratch

<p align="center">
  <img src="./myllm.png" width="800" alt="MyLLM Overview">
</p>

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Tests](https://img.shields.io/badge/tests-128%20passed-brightgreen.svg)

---

## What is MyLLM?

**MyLLM** is a from-scratch LLM framework built for deep understanding and real research.

It covers the full pipeline:

> **Tokenization → Attention → Training → RLHF → Inference**

There are already great libraries (HuggingFace, Lightning, TRL). But they hide too much.

MyLLM is intentionally different:

- **Pure PyTorch** — no HuggingFace model abstractions in the core
- **Every line visible** — no hidden magic, no black boxes
- **Research-friendly** — SFT, DPO, PPO, quantization
- **From scratch** — so you *actually* understand what's happening

---

## Architecture

MyLLM is structured as a **learning → experimentation → framework** pipeline.

```
MyLLM/
├── notebooks/        # 21 guided notebooks — learn by doing
├── Modules/          # Isolated experiments — one concept at a time
├── demos/            # 5 Colab-ready notebooks (install → quickstart → SFT)
├── docs/             # Component-by-component reference documentation
└── myllm/            # ⭐ Core framework — installable package
```

---

## `myllm/` — The Core Framework

A clean, installable LLM framework — pure PyTorch, fully transparent.

```
myllm/
├── model.py          # GPT / LLaMA-style transformer
├── api.py            # LLM — from_pretrained, generate, generate_text, generate_batch
├── Configs/          # ModelConfig, GenerationConfig
├── Tokenizers/       # GPT2 (tiktoken), LLaMA2 (SentencePiece), LLaMA3, trainable
├── Train/
│   ├── sft_trainer.py        # Supervised Fine-Tuning ✅
│   ├── dpo_trainer.py        # Direct Preference Optimization (in progress)
│   ├── ppo_trainer.py        # PPO / RLHF (in progress)
│   └── Engine/               # Training loop, accelerators, callbacks
│       └── accelerator/      # Single GPU, DDP, DeepSpeed, FSDP
└── utils/            # ModelLoader, weight mappers, sampler, model registry
```

### Install

```bash
# From GitHub (always latest)
pip install git+https://github.com/silvaxxx1/MyLLM.git

# Local editable install
git clone https://github.com/silvaxxx1/MyLLM.git
cd MyLLM
pip install -e .          # or: uv pip install -e .

# With optional groups
pip install "myllm[train] @ git+https://github.com/silvaxxx1/MyLLM.git"    # + wandb, accelerate, deepspeed
pip install "myllm[inference] @ git+https://github.com/silvaxxx1/MyLLM.git" # + matplotlib, pandas, seaborn
pip install "myllm[all] @ git+https://github.com/silvaxxx1/MyLLM.git"       # everything
```

### Import styles — three ways, all equivalent

```python
# Flat top-level (most convenient)
from myllm import LLM, ModelConfig, GenerationConfig
from myllm import SFTTrainer, SFTTrainerConfig
from myllm import get_tokenizer

# Submodule style
from myllm.train import SFTTrainer, SFTTrainerConfig
from myllm.tokenizers import GPT2Tokenizer, get_tokenizer
from myllm.configs import ModelConfig, GenerationConfig

# Attribute access
import myllm
myllm.train.SFTTrainer
myllm.tokenizers.GPT2Tokenizer
```

### CLI

```bash
python -m myllm version          # myllm 0.1.0
python -m myllm models           # list all available model configs
python -m myllm info gpt2-medium # layers, heads, params, memory estimate
```

### Load a model and generate

```python
from myllm import LLM, GenerationConfig

# Config + weights + tokenizer — one call
llm = LLM.from_pretrained("gpt2-small")
print(llm)
# LLM(model='gpt2-small', params=124.4M, device='cuda', dtype=torch.float32)

result = llm.generate_text(
    "The future of AI is",
    generation_config=GenerationConfig(max_length=60, temperature=0.8, top_k=50),
    skip_prompt=True,   # return only the generated text, not the input prompt
)
print(result["text"])
```

### Fine-tune with SFT

```python
from myllm import ModelConfig
from myllm.train import SFTTrainer, SFTTrainerConfig

trainer = SFTTrainer(
    SFTTrainerConfig(output_dir="./output", num_epochs=3, report_to=[]),
    model_config=ModelConfig.from_name("gpt2-small"),
)
trainer.setup_model()
trainer.setup_data(train_dataloader=my_dataloader)
trainer.setup_optimizer()
trainer.train()
```

### Run tests

```bash
uv run pytest       # 128 tests, ~15s on CPU, no GPU or weights required
```

---

## Test Suite

All tests run against a tiny randomly-initialised model (2 layers / 64 dim) — no pretrained weights, CPU-only.

```
myllm/tests/
├── conftest.py          # shared fixtures and batch utilities
├── test_config.py       # ModelConfig, GenerationConfig
├── test_model.py        # GPT: MLP variants, KV cache, attention, RoPE, forward pass
├── test_tokenizers.py   # tokenizer factory, GPT-2 encode/decode, wrapper
├── test_api.py          # LLM: generate, generate_text, generate_batch, all sampling modes
├── test_sampler.py      # repetition penalty, top-k, top-p, EOS detection
├── test_training.py     # SFT / Pretrain / Classifier trainers, configs, checkpoints
└── test_e2e.py          # full pipeline: init → train → checkpoint → generate
```

**128 passed — ~15s on CPU:**

| Module | Tests | What's covered |
|--------|-------|----------------|
| Config | 14 | Presets, validation, save/load, memory estimation |
| Model | 20 | MLP variants, KV cache, RMSNorm, attention shapes, RoPE |
| Tokenizers | 16 | Factory caching, encode/decode, special tokens, padding |
| API | 19 | generate(), generate_text(), generate_batch(), sampling modes |
| Sampler | 15 | Repetition penalty, top-k, top-p, EOS detection |
| Training | 36 | All trainers, configs, datasets, checkpoint save/load |
| E2E | 8 | Init → train → checkpoint → inference |

---

## `notebooks/` — Learn by Doing

21 notebooks. Theory → implementation → experimentation, in order.

```
0.0  Welcome & orientation
1.1  Data & tokenization
1.2  Byte-pair encoding from scratch
2.1  Attention from scratch
2.2  Multi-head, grouped-query, flash attention
2.3  GPT architecture
2.4  LLaMA 3 architecture
3.1  Training loop
3.2  Advanced training (AMP, grad accumulation, distributed)
4.1  SFT — text classification
4.2  SFT — instruction following
4.3  PEFT / LoRA
5.1  RLHF with PPO
5.2  DPO
6.1  Inference & text generation
6.2  KV cache
6.3  Quantization (INT8)
6.4  Quantization (INT4 / GPTQ)
A/B  Appendices: GPT-2 vs LLaMA2, Gradio UI
```

---

## `Modules/` — Targeted Experiments

One concept per module. Proving ground before ideas graduate into the core framework.

```
Modules/
├── 1.data/        # Dataset loading & preprocessing
├── 2.models/      # GPT, LLaMA architectures, attention variants (MHA/MQA/GQA/Flash)
├── 3.training/    # Training loops and utilities
├── 4.finetuning/  # SFT, DPO, PPO, QLoRA experiments
└── 5.inference/   # KV cache, quantization
```

---

## `demos/` — Try it on Colab

Five Colab-ready notebooks. Each installs `myllm` automatically.

| Notebook | What it covers |
|----------|---------------|
| `00_install_and_setup.ipynb` | Install, import styles, CLI, memory estimates |
| `01_quickstart.ipynb` | `from_pretrained` → generate → `skip_prompt` |
| `02_generation_configs.ipynb` | All sampling strategies compared |
| `03_tokenizers_and_configs.ipynb` | Tokenizers, ModelConfig, save/load |
| `04_sft_training.ipynb` | Fine-tune GPT-2 on an instruction dataset end-to-end |

---

## `docs/` — Reference Documentation

34-page component-by-component documentation covering every public class and method.

```
docs/
├── getting-started/   # installation, quickstart
├── core/              # ModelConfig, GenerationConfig, LLM, GPT model
├── tokenizers/        # overview, factory, wrapper, GPT2, LLaMA2, LLaMA3, trainable
├── training/          # overview, BaseTrainer, SFTTrainer, PretrainTrainer, DPO, PPO
├── training-configs/  # TrainerConfig, SFTConfig, DPOConfig
├── engine/            # TrainerEngine, accelerators, callbacks, checkpointing
├── utils/             # ModelLoader, ModelRegistry, WeightMappers, OptimizedSampler
├── cli.md
├── testing.md
└── extension-guide.md
```

Start at [`docs/index.md`](docs/index.md).

---

## Supported Models

Weights are auto-downloaded from HuggingFace on first `LLM.from_pretrained()` call and cached in `./models/`.

| Model | Params | Auth required | Min VRAM (fp16) |
|-------|--------|---------------|-----------------|
| `gpt2-small` | 124M | — | < 1 GB |
| `gpt2-medium` | 335M | — | 1 GB |
| `gpt2-large` | 774M | — | 2 GB |
| `gpt2-xl` | 1.5B | — | 4 GB |
| `llama3-1b` | 1.9B | HF token | 3 GB |
| `llama3-8b` | 8B | HF token | 17 GB |
| `llama2-7b` | 7B | HF token + license | 16 GB |
| `llama2-13b` | 13B | HF token + license | 32 GB |

---

## Roadmap

| Status | Item |
|--------|------|
| ✅ | 21 learning notebooks (tokenization → RLHF → inference) |
| ✅ | Modular experiments (GPT, LLaMA, attention variants, SFT, DPO, PPO, quantization) |
| ✅ | Installable `myllm` package with public API |
| ✅ | GPT-2 / LLaMA-2 / LLaMA-3 loading + generation |
| ✅ | `LLM.from_pretrained()` — one-line model + tokenizer loader |
| ✅ | SFT Trainer (AMP, gradient accumulation, WandB, checkpointing) |
| ✅ | 5 Colab-ready demo notebooks |
| ✅ | 34-page component-level documentation |
| ✅ | 128-test suite (CPU-only, no weights needed) |
| 🔧 | DPO Trainer — implementation in progress |
| 🔧 | PPO Trainer — implementation in progress |
| ⬜ | Streaming generation (`generate_stream()`) |
| ⬜ | ModelConfig entries for Mistral, Phi-2, Gemma |
| ⬜ | 8-bit / 4-bit quantization in core `myllm/` |

---

## Setup

```bash
# Recommended
uv sync

# Or pip
pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.x

---

## Inspiration

- **Andrej Karpathy** — NanoGPT minimalism
- **Umar Jamil** — Practical transformer intuition
- **Sebastian Raschka** — Deep theoretical clarity

---

## License

MIT — see `LICENSE` for details.
