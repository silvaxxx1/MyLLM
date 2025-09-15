myllm/training/
├── __init__.py
├── configs.py              # Training-specific configurations
├── base.py                 # BaseTrainer abstract class
├── pretrainer.py          # Pretraining (Causal LM)
├── sft_trainer.py         # Supervised Fine-tuning
├── adapters/
│   ├── __init__.py
│   ├── base.py            # BaseAdapter interface
│   ├── lora.py            # LoRA implementation
│   └── qlora.py           # QLoRA implementation
├── rlhf/
│   ├── __init__.py
│   ├── dpo_trainer.py     # Direct Preference Optimization
│   └── grpo_trainer.py    # Group Relative Policy Optimization
├── utils/
│   ├── __init__.py
│   ├── data.py            # Data utilities and collators
│   ├── metrics.py         # Training metrics
│   ├── callbacks.py       # Training callbacks
│   └── checkpoint.py      # Checkpoint management
└── datasets/
    ├── __init__.py
    ├── pretrain.py        # Pretraining datasets
    ├── sft.py             # SFT datasets
    └── preference.py      # RLHF preference datasets