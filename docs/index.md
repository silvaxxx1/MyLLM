# myllm Documentation

> A from-scratch LLM framework: tokenization → attention → training → RLHF → inference.
> Pure PyTorch. No HuggingFace model wrappers. Every line maps to real code.

---

## Navigation

### Getting Started
- [Installation](getting-started/installation.md) — pip, uv, Colab, editable install
- [Quickstart](getting-started/quickstart.md) — load a model and generate in 2 lines

### Core
- [ModelConfig](core/model-config.md) — architecture configuration for all model families
- [GenerationConfig](core/generation-config.md) — sampling, temperature, KV cache
- [LLM](core/llm.md) — high-level inference wrapper
- [GPT Model](core/gpt-model.md) — transformer architecture internals

### Tokenizers
- [Overview](tokenizers/overview.md) — design, factory, wrapper interface
- [GPT2Tokenizer](tokenizers/gpt2.md) — tiktoken BPE (GPT-2/3/4 families)
- [LLaMA2Tokenizer](tokenizers/llama2.md) — SentencePiece
- [LLaMA3Tokenizer](tokenizers/llama3.md) — tiktoken cl100k
- [TrainableTokenizer](tokenizers/trainable.md) — train BPE from scratch
- [TokenizerWrapper](tokenizers/wrapper.md) — unified interface
- [Factory](tokenizers/factory.md) — `get_tokenizer()`, registration, caching

### Training
- [Overview](training/overview.md) — trainer hierarchy and lifecycle
- [BaseTrainer](training/base-trainer.md) — shared training loop, AMP, optimizer, checkpointing
- [SFTTrainer](training/sft-trainer.md) — supervised fine-tuning with response masking
- [PretrainTrainer](training/pretrain-trainer.md) — next-token prediction pretraining
- [DPOTrainer](training/dpo-trainer.md) — direct preference optimization *(stub)*
- [PPOTrainer](training/ppo-trainer.md) — reinforcement learning from human feedback *(stub)*

### Training Configs
- [TrainerConfig](training-configs/trainer-config.md) — base config with all common fields
- [SFTTrainerConfig](training-configs/sft-config.md) — SFT-specific fields
- [DPOTrainerConfig](training-configs/dpo-config.md) — DPO-specific fields

### Training Engine
- [TrainerEngine](engine/trainer-engine.md) — generic composable training loop
- [Accelerators](engine/accelerators.md) — single GPU, DDP, DeepSpeed, FSDP
- [Callbacks](engine/callbacks.md) — hook system for training events
- [CheckpointManager](engine/checkpointing.md) — save, load, prune checkpoints

### Utilities
- [ModelLoader](utils/model-loader.md) — download, cache, map weights
- [ModelRegistry](utils/model-registry.md) — supported model families and variants
- [WeightMappers](utils/weight-mappers.md) — HuggingFace → myllm key translation
- [OptimizedSampler](utils/sampler.md) — top-k, top-p, repetition penalty

### Reference
- [CLI](cli.md) — `python -m myllm` commands
- [Testing](testing.md) — test suite structure and fixtures
- [Extension Guide](extension-guide.md) — add models, tokenizers, trainers

---

## Project Layout

```
MyLLM/
├── myllm/              # installable package (pip install -e .)
│   ├── model.py        # GPT transformer
│   ├── api.py          # LLM inference wrapper
│   ├── Configs/        # ModelConfig, GenerationConfig
│   ├── Tokenizers/     # tokenizer implementations + factory
│   ├── Train/          # trainers, configs, engine
│   └── utils/          # loader, registry, sampler
├── demos/              # 5 Colab-ready notebooks
├── notebooks/          # 21 guided learning notebooks
├── Modules/            # isolated experiments
└── docs/               # this documentation
```

## Three-Layer Architecture

| Layer | Purpose | Audience |
|-------|---------|---------|
| `notebooks/` + `Modules/` | Learning from first principles | Students, researchers |
| `demos/` | End-to-end usage examples | Users of the library |
| `myllm/` | Production-grade core | Library consumers, contributors |

## Package Version

Current: `0.1.0`
