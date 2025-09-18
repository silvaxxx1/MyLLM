
# README.md
"""
# ML Training Framework

A training framework designed to integrate seamlessly with your existing transformer project architecture.

## Integration with Existing Project

This framework is specifically designed to work with your existing:
- `model.py`: GPT transformer implementation
- `ModelConfig`: Model configuration management  
- `api.py`: LLM wrapper for generation
- Tokenizer factory system

## Key Features

### Seamless Integration
- Uses your existing `ModelConfig` for model parameters
- Compatible with your `GPT` model implementation
- Integrates with your tokenizer factory
- Maintains compatibility with your `LLM` API

### Training Capabilities
- **Pre-training**: Standard language model training
- **SFT**: Supervised fine-tuning with instruction formatting
- **PPO/DPO**: RLHF training (coming soon)

### Advanced Logging
- WandB integration with automatic model artifact logging
- TensorBoard support
- Comprehensive metrics tracking
- Training progress monitoring

## Quick Start

```python
from trainer import Trainer, TrainerConfig
from Configs import ModelConfig

# Create training config
config = TrainerConfig(
    model_config_name="gpt2-small",  # Uses your ModelConfig
    tokenizer_name="gpt2",           # Uses your tokenizer factory
    wandb_project="my-training"
)

# Train model
trainer = Trainer(config)
trainer.train()
```

## Configuration

The framework uses your existing `ModelConfig` and extends it with training-specific settings:

```python
# Your existing model config is used automatically
model_config = ModelConfig.from_name("gpt2-small")

# Training config adds training-specific parameters
trainer_config = TrainerConfig(
    model_config_name="gpt2-small",
    learning_rate=5e-5,  # Overrides ModelConfig if specified
    batch_size=8,
    num_epochs=3
)
```

## Using Trained Models

Trained models are fully compatible with your existing `api.py`:

```python
from api import LLM
from Configs import ModelConfig

# Load your trained model
llm = LLM(config=ModelConfig.from_name("gpt2-small"))

# Load trained weights instead of pretrained
checkpoint_path = "./output/checkpoint-1000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path)
llm.model.load_state_dict(state_dict)

# Use your existing generation pipeline
result = llm.generate_text(prompt, tokenizer, gen_config)
```

## Architecture Compatibility

- **Model**: Uses your `GPT` class from `model.py`
- **Config**: Extends your `ModelConfig` system
- **Tokenization**: Uses your tokenizer factory
- **Generation**: Compatible with your `LLM` wrapper
- **Checkpoints**: Saves in format compatible with your `api.py`

This framework enhances your existing architecture without requiring changes to your core model implementation.
"""