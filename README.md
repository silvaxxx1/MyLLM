# 🚀 MyLLM — A Transparent LLM Framework, Built From Scratch

<p align="center">
  <img src="./myllm.png" width="800" alt="MyLLM Overview">
</p>

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)

---

## What is MyLLM?

**MyLLM** is a from-scratch LLM framework built for deep understanding and real research.

It covers the full pipeline:

> **Tokenization → Attention → Training → RLHF → Inference**

The core goal is simple:

> **Understand every single line of a modern transformer stack —**  
> **then build a clean, research-grade framework around it.**

There are already great libraries (🤗 Hugging Face, Lightning, TRL…). But they hide too much.

MyLLM is intentionally different:

- **Minimal** — no unnecessary abstractions
- **Hackable** — everything is visible and editable
- **Research-friendly** — LoRA, QLoRA, PPO, DPO, quantization
- **From scratch** — so you *actually* understand what's happening

---

## 🏗 Architecture

MyLLM is structured as a **learning → experimentation → framework** pipeline.

```
MyLLM/
├── notebooks/        # 21 guided notebooks — learn by doing
├── Modules/          # Isolated experiments — one concept at a time
└── myllm/            # ⭐ Core framework — the main focus
```

---

## ⭐ `myllm/` — The Core Framework

This is the heart of the project. A **HuggingFace-like framework, but fully transparent** — pure PyTorch, no black boxes.

```
myllm/
├── model.py          # Core LLM definition (GPT / LLaMA-style)
├── api.py            # REST API for serving models (FastAPI)
├── Configs/          # Centralized model & generation configs
├── Tokenizers/       # GPT2, LLaMA2, LLaMA3, trainable tokenizer
├── Train/            # Training engines
│   ├── sft_trainer.py        # Supervised Fine-Tuning
│   ├── dpo_trainer.py        # Direct Preference Optimization
│   ├── ppo_trainer.py        # PPO / RLHF
│   └── Engine/               # Training loop, accelerators, callbacks
│       └── accelerator/      # Single GPU, DDP, DeepSpeed, FSDP
└── utils/            # Loaders, samplers, weight mappers
```

### Quick Start

```python
from myllm.model import LLMModel
from myllm.Train.sft_trainer import SFTTrainer

model = LLMModel()

trainer = SFTTrainer(
    model=model,
    dataset=my_dataset
)

trainer.train()
```

### Serve via REST API

```bash
python myllm/api.py
```

### Run Tests

```bash
uv run pytest
```

> Every line maps to **real code**. No hidden magic. No black boxes.

---

## 🧪 Test Suite

All tests live in `myllm/tests/` and run against a tiny randomly-initialised model (2-layer / 64-dim) — no pretrained weights required, CPU-only.

```
myllm/tests/
├── conftest.py          # Shared fixtures and batch utilities
├── test_config.py       # ModelConfig & GenerationConfig
├── test_model.py        # GPT components: MLP, KVCache, Attention, RoPE, forward pass
├── test_tokenizers.py   # Tokenizer factory, GPT-2 encode/decode
├── test_api.py          # LLM wrapper: generate, generate_text, generate_batch
├── test_sampler.py      # Repetition penalty, top-k, top-p, EOS detection
├── test_training.py     # Trainers (Pretrain / SFT / Classifier), configs, factory, datasets
└── test_e2e.py          # Full pipeline: init → train → checkpoint → generate
```

**Results (128 tests, 48s on CPU + GPU):**

```
======================== 128 passed in 48.24s ==========================
```

| Module | Tests | Coverage |
|--------|-------|----------|
| Config | 14 | ModelConfig presets, validation, save/load, memory estimation |
| Model | 20 | MLP variants, KVCache, RMSNorm, Attention shapes, RoPE, full forward |
| Tokenizers | 16 | Factory caching, GPT-2 encode/decode, special tokens, Unicode |
| API | 19 | generate(), generate_text(), generate_batch(), all sampling modes |
| Sampler | 15 | Repetition penalty, top-k, top-p, EOS detection |
| Training | 36 | All 3 trainers, configs, datasets, factory, checkpoint save/load |
| E2E | 8 | Init → train steps → checkpoint → inference → full pipeline |

---

## 1️⃣ `notebooks/` — Learn by Doing

The entry point for understanding. Each notebook explains theory, implements from scratch, and encourages experimentation.

```
notebooks/
├── 0.0.WELCOME.ipynb
├── 1.1.DATA.ipynb
├── 1.2.Tokenizer.ipynb
├── 2.1.ATTENTION.ipynb
├── 2.2.More_ATTENTION.ipynb
├── 2.3.GPT.ipynb
├── 2.4.Llama3.ipynb
├── 3.1.TRAIN.ipynb
├── 3.2.TRAIN_Pro.ipynb
├── 4.1.SFT_Text_Classification.ipynb
├── 4.2.SFT_Instruction_Following.ipynb
├── 4.3.SFT_PEFT.ipynb
├── 5.1.RLHF_PPO.ipynb
├── 5.2.RL_DPO.ipynb
├── 6.1.INFERENCE_Text_Generation.ipynb
├── 6.2.KV_Cache.ipynb
├── 6.3.Quantization_1.ipynb
├── 6.4.Quantization_2.ipynb
└── Appendices (GPT-2/LLaMA2, Gradio UI)
```

💡 *Change an attention mask and immediately see how generation breaks or improves. That's real learning.*

---

## 2️⃣ `Modules/` — Targeted Experiments

Isolates one concept at a time. A proving ground before ideas graduate into the core framework.

```
Modules/
├── 1.data/        # Dataset loading & preprocessing
├── 2.models/      # GPT, LLaMA-style architectures, attention variants
├── 3.training/    # Training loops & utilities
├── 4.finetuning/  # SFT, DPO, PPO experiments
└── 5.inference/   # KV cache, quantization
```

### Example: Train a small GPT from scratch

```bash
python Modules/3.training/train.py --config configs/basic.yml
```

---

## ⚙️ Setup

```bash
uv sync
```

**Requirements:** Python 3.10+, PyTorch 2.x, CUDA recommended.

---

## 📦 Pre-trained Weights

GPT-2 weights are included for experimentation:

| Model | Parameters | File |
|-------|-----------|------|
| GPT-2 Small | 124M | `models/model-gpt2-small.safetensors` |
| GPT-2 Medium | 335M | `models/model-gpt2-medium.safetensors` |
| GPT-2 Large | 774M | `models/model-gpt2-large.safetensors` |
| GPT-2 XL | 1.5B | `models/model-gpt2-xl.safetensors` |

---

## 📍 Roadmap

| Status | Milestone | Description |
|--------|-----------|-------------|
| ✅ | Interactive Notebooks | From-first-principles learning |
| ✅ | Modular Mini-Projects | Reusable experiments |
| ⚙️ | **`myllm` Core Framework** | **SFT, DPO, PPO, quantization — active development** |

---

## ⚡ Quick Challenges

- Modify attention masks and observe behavior
- Train a GPT on a custom dataset
- Add a new trainer (TRL-style) to `myllm/Train/`
- Quantize a model and benchmark speed
- Implement a new attention variant and benchmark it in `Modules/2.models/atten/`

---

## 🙌 Inspiration

- **Andrej Karpathy** — NanoGPT minimalism
- **Umar Jamil** — Practical transformer intuition
- **Sebastian Raschka** — Deep theoretical clarity

---

## 🏁 The Vision

A **transparent, educational, and production-ready LLM framework**  
built from scratch by people who want to **own every line of their AI system**.

Let's remove the black boxes and **build LLMs the right way**.

---

## 📜 License

MIT License — see `LICENSE` for details.