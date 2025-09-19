# 🏋️ MyLLM Train — The Training Engine of MyLLM Core

Welcome to **MyLLM Train**, the **heart of the MyLLM project**, where **models are built, optimized, and fine-tuned**.

This submodule is designed to provide a **lightweight yet powerful training backend** for **LLM development**, offering deep control over every step of the training process while staying minimal and hackable.

> **Philosophy:**
> Instead of hiding complexity like other frameworks, MyLLM Train **exposes the internals** — so you can **modify, inspect, and truly understand** what’s happening under the hood.

---

## 🔹 What This Engine Does

The `Train` module is the **training backbone** of the MyLLM ecosystem.
It supports:

* ✅ **Core Training Loops** — for supervised fine-tuning (SFT), DPO, PPO, and custom algorithms.
* ✅ **Accelerator Management** — single GPU, multi-GPU (DDP/FSDP), Hugging Face Accelerate, or DeepSpeed.
* ✅ **Optimizers & Schedulers** — plug-and-play optimizer configs, LR schedulers, and gradient clipping.
* ✅ **Checkpointing** — robust saving, resuming, and experiment tracking.
* ✅ **Callbacks System** — hooks for logging, metrics, and custom behaviors.
* ✅ **Config-Driven Design** — YAML/JSON configs for fully reproducible experiments.

---

## 🏗 Folder Structure

Here’s how the `myllm/Train` subproject is organized:

```bash
myllm/Train/
├── base_trainer.py            # Abstract base class for all trainers
├── trainer.py                  # Default trainer implementation
├── sft_trainer.py              # Supervised Fine-Tuning trainer
├── dpo_trainer.py              # Direct Preference Optimization trainer
├── ppo_trainer.py              # PPO trainer for RLHF
├── train_test.py               # Training tests and experiments
│
├── configs/                    # Configuration management
│   ├── TrainerConfig.py        # Core training configs
│   ├── SFTConfig.py            # Configs for SFT runs
│   ├── PPOConfig.py            # Configs for PPO runs
│   ├── DPOConfig.py            # Configs for DPO runs
│   ├── TempConfig.py           # Temporary or experimental configs
│   └── training_config.yaml    # Example YAML config
│
├── Engine/                      # Core training engine
│   ├── accelerator/             # Device and backend handling
│   │   ├── base.py              # Abstract accelerator interface
│   │   ├── single_gpu.py        # Single GPU training
│   │   ├── ddp_accelerate.py    # PyTorch Distributed Data Parallel (DDP)
│   │   ├── fspd_accelerate.py   # Fully Sharded Data Parallel (FSDP)
│   │   ├── hf_accerlerate.py    # Hugging Face Accelerate wrapper
│   │   └── deepspeed_accerlerate.py  # DeepSpeed integration
│   │
│   ├── optimizer.py             # Optimizer management
│   ├── lr_scheduler.py          # Learning rate schedulers
│   ├── checkpoint_manager.py    # Save/load checkpoints
│   ├── callbacks.py             # Custom callbacks & hooks
│   ├── trainer_engine.py        # Core engine orchestration
│   ├── utils.py                  # Utility functions
│   └── test/                     # Unit tests
│       ├── test_acceletrate.py
│       └── test_optimizer.py
│
├── utils/                        # General utilities
│   ├── config_manager.py
│   ├── logging_utils.py
│   └── __init__.py
│
└── structure.md                   # Documentation of internal design
```

---

## ⚙️ Core Design

At a high level, the engine is divided into **four core layers**:

| Layer           | Role                                                       | Example Class/Module       |
| --------------- | ---------------------------------------------------------- | -------------------------- |
| **Trainer**     | Implements a training strategy (SFT, PPO, DPO, etc.)       | `SFTTrainer`, `PPOTrainer` |
| **Engine**      | Orchestrates the training loop & integrates all components | `TrainerEngine`            |
| **Accelerator** | Abstracts away device/backend differences                  | `SingleGPUAccelerator`     |
| **Managers**    | Manages optimizers, schedulers, checkpoints, configs       | `OptimizerManager`         |

This separation gives **clear boundaries** and makes it easy to extend:

* Want to try a new distributed backend? → Implement a new `Accelerator`.
* Want to add a new RL algorithm? → Extend `BaseTrainer` and plug into the engine.
* Want custom logging? → Add a callback.

---

## 🚀 Supported Training Backends

The engine supports multiple distributed backends **out of the box**, each with its own use case:

| Backend           | File                       | Use Case                                        |
| ----------------- | -------------------------- | ----------------------------------------------- |
| **Single GPU**    | `single_gpu.py`            | Simple experiments or debugging                 |
| **DDP**           | `ddp_accelerate.py`        | Multi-GPU training with minimal overhead        |
| **FSDP**          | `fspd_accelerate.py`       | Extremely large models with sharded memory      |
| **HF Accelerate** | `hf_accerlerate.py`        | Quick experiments, integrates with HF ecosystem |
| **DeepSpeed**     | `deepspeed_accerlerate.py` | Very large models with ZeRO optimizations       |

**Tip:**
You don't *need* Hugging Face Accelerate unless you want to leverage its ecosystem (e.g., fast prototyping, HF `Trainer` integration).
For production, DDP/FSDP or DeepSpeed are preferred.

---

## 🧩 Key Components

### 1. **Base Trainer (`base_trainer.py`)**

The `BaseTrainer` defines the **interface every trainer must implement**, such as:

* `train_dataloader()`
* `train_step()`
* `batch_to_device()`
* `evaluate()`

Each specialized trainer (e.g., `SFTTrainer`, `PPOTrainer`) inherits from this and adds domain-specific logic.

---

### 2. **Trainer Engine (`trainer_engine.py`)**

The `TrainerEngine` is the **conductor** of the training process:

* Calls `trainer.train_step()` on each batch
* Handles gradient accumulation and clipping
* Coordinates between the accelerator, optimizer, and callbacks
* Saves checkpoints automatically

Think of it as the **PyTorch Lightning Trainer**, but lightweight and fully transparent.

---

### 3. **Accelerators**

Located in `Engine/accelerator/`, each accelerator encapsulates a specific backend.
This allows the **same trainer code** to run on:

* Single GPU
* Multi-GPU (DDP/FSDP)
* Hugging Face Accelerate
* DeepSpeed

Example:

```python
from myllm.Train.Engine.accelerator import SingleGPUAccelerator

acc = SingleGPUAccelerator(config)
```

Switching to FSDP:

```python
from myllm.Train.Engine.accelerator import FSDPAccelerator

acc = FSDPAccelerator(config)
```

---

### 4. **Managers**

Reusable components for:

* Optimizers (`OptimizerManager`)
* Learning rate schedulers (`SchedulerManager`)
* Checkpoints (`CheckpointManager`)

Each manager is **plug-and-play**, making experiments more modular.

---

### 5. **Callbacks**

Callbacks allow you to **hook into the training lifecycle**:

* Logging
* Metrics collection
* Custom debugging
* External integrations (e.g., WandB, TensorBoard)

Example:

```python
from myllm.Train.Engine.callbacks import PrintCallback

engine = TrainerEngine(
    trainer=my_trainer,
    callbacks=[PrintCallback()],
)
```

---

## 🔗 Example Training Script

```python
# Pseudocode for training
from myllm.Train.Engine.accelerator import SingleGPUAccelerator
from myllm.Train.Engine.trainer_engine import TrainerEngine
from myllm.Train.Engine.optimizer import OptimizerManager
from myllm.Train.Engine.lr_scheduler import SchedulerManager
from myllm.Train.Engine.checkpoint_manager import CheckpointManager
from myllm.Train.Engine.callbacks import PrintCallback
from myllm.Train.sft_trainer import SFTTrainer

trainer = SFTTrainer(model=my_model, dataset=my_dataset)

config = {
    "num_epochs": 3,
    "gradient_clip": 1.0,
    "optimizer": {"name": "adamw", "lr": 1e-4}
}

acc = SingleGPUAccelerator(config)
opt_mgr = OptimizerManager(trainer.model, config)
sched_mgr = SchedulerManager(opt_mgr.optimizer, config)
ckpt_mgr = CheckpointManager(trainer.model)

engine = TrainerEngine(
    trainer, acc, opt_mgr, scheduler_manager=sched_mgr,
    checkpoint_manager=ckpt_mgr,
    callbacks=[PrintCallback()],
    config=config
)

engine.setup()
engine.train()
```

---

## 🔮 Future Plans

| Feature                        | Status         |
| ------------------------------ | -------------- |
| Mixed precision training (AMP) | ⚙️ In progress |
| TPU support via XLA            | 🛠 Planned     |
| Native WandB integration       | 🛠 Planned     |
| LoRA & QLoRA integration       | 🛠 Planned     |
| Dataset streaming support      | 🛠 Planned     |

---

## 🧠 Why This Matters

Most frameworks (Hugging Face, Lightning, etc.) are **too abstracted**, making it difficult to:

* Debug deep internals
* Experiment with custom algorithms
* Scale efficiently without vendor lock-in

MyLLM Train **solves this** by being:

* **Lightweight**: No extra dependencies, pure PyTorch core.
* **Transparent**: Every line of code is accessible and hackable.
* **Scalable**: From your laptop → multi-GPU cluster → massive distributed training.

---

## 📜 License

MIT License — use it, break it, and make it better.

---

## 🤝 Contribution

Contributions are welcome!
Whether it's:

* Adding a new trainer
* Building a new accelerator
* Improving documentation

Just fork, hack, and submit a PR.

---

## 🚀 Final Note

MyLLM Train isn’t just a training engine — it’s a **learning tool** and **research platform**.

> **The goal:** Give you the **tools and visibility** to build, understand, and scale LLM training **without black boxes**.

Go ahead — **train your own MetaBot**. 🦾
