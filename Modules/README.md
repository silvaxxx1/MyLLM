# 🔩 MyLLM: Modular Framework

## *Structured Components for End-to-End LLM Development*

The `Modules` directory is the **core playground** of MyLLM — a fully modular and transparent framework for **building, training, fine-tuning, and deploying LLMs**.

Think of this as a **Hugging Face alternative**, but designed for:

* 🔍 **Clarity** – Clean, hackable code for deep understanding.
* 🧪 **Research-first workflows** – Perfect for experimenting with new ideas.
* ⚡ **Modularity** – Swap in/out components without breaking the whole pipeline.

---

## 📂 Directory Overview

```
Modules/
│
├── 1.data/         # Data loading, preprocessing, tokenization
├── 2.models/       # Core model architectures (GPT, LLaMA, custom)
├── 3.training/     # Training loops, distributed training, utilities
├── 4.finetuning/   # LoRA, QLoRA, RLHF, DPO, custom finetuning scripts
├── 5.inference/    # Efficient inference, quantization, deployment
└── README.md       # This file
```

Each folder represents a **self-contained stage** in the LLM lifecycle.

---

## 🧩 Module Details

### **1. Data**

> *"Garbage in, garbage out."*
> This module handles **all data-related workflows**:

* Loading raw datasets (text, JSON, CSV)
* Preprocessing & cleaning
* Tokenization (custom or Hugging Face)
* Creating binary dataset splits (`train_ids.bin`, `val_ids.bin`)

**Key Files:**

* `dataloader.py` – Dataset loading logic
* `preprocess.py` – Data cleaning and prep
* `Tokenizer/` – Custom tokenizer implementations
* `tests/` – Unit tests for data pipelines

---

### **2. Models**

Centralized **model architectures**:

* GPT family (`GPT2`, `GPT-XL`)
* LLaMA family
* Custom attention mechanisms (Flash Attention, MQA, GQA)

**Key Folders:**

* `GPT/` – Standard GPT architectures
* `LLAMA/` – LLaMA variants
* `atten/` – Experimental attention modules

---

### **3. Training**

Core **training logic**, designed to be scalable and minimal:

* Single-GPU and multi-GPU training (`train.py`, `train_dist.py`)
* Distributed training utilities
* Modular `Trainer` class
* Clean, reproducible workflows

**Key Files:**

* `train.py` – Standard training entry point
* `train_dist.py` – Distributed training
* `trainer.py` – Encapsulated training loop
* `train_utils.py` – Reusable training utilities

---

### **4. Fine-Tuning**

Purpose-built for **specialized model adaptation**:

* LoRA & QLoRA
* RLHF (Reinforcement Learning with Human Feedback)
* DPO (Direct Preference Optimization)
* Instruction-tuned models (e.g., Alpaca)

**Key Folders:**

* `GPT2_124M_SPAM/` – Small-scale experiments
* `GPT_XL_ALPACA/` – Instruction-tuning setups
* `GPT2_RLHF_PPO/` – PPO for RLHF
* `GPT2_RL_DPO/` – Direct Preference Optimization

---

### **5. Inference**

Optimized inference and deployment:

* Model quantization
* Efficient GPT inference pipelines
* Ready-to-deploy scripts for production

**Key Folders:**

* `GPT2_Inference/` – GPT inference setup
* `GPT_Quantizer/` – Quantization utilities
* `requirements.txt` – Minimal dependencies

---

## 🗺️ MyLLM Roadmap

This modular framework works alongside **learning notebooks** to create a full-stack LLM journey.

1. **Learning Phase (Notebooks)**

   * Transformers from scratch
   * Flash Attention, MQA, GQA
   * LoRA & QLoRA
   * RLHF with PPO & DPO
   * KV caching & quantization
   * *And more coming soon…*

   📒 [Notebooks](https://lnkd.in/drkeKUre)

2. **Modular Playground**

   * Targeted experiments using these modules.
   * Each module can be extended or replaced for custom research.

   🧩 [Playground Repo](https://lnkd.in/dQX9Dhi4)

3. **Core Framework**

   * A minimal, transparent pipeline — research-first, fully hackable.

   ⚙️ [Core Repo](https://lnkd.in/dBR5Th6w)

---