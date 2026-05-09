"""
Shared fixtures for the MyLLM test suite.

All fixtures use a tiny 2-layer / 64-dim model so tests run fast
on CPU without any pretrained weights.
"""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from myllm.Configs import ModelConfig, GenerationConfig
from myllm.model import GPT


# ---------------------------------------------------------------------------
# Model / Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_model_config() -> ModelConfig:
    """Minimal GPT config: 2 layers, 64 dim, 2 heads, vocab 1000, ctx 32."""
    return ModelConfig(
        n_layer=2,
        n_embd=64,
        n_head=2,
        vocab_size=1000,
        block_size=32,
        mlp_class_name="GptMLP",
    )


@pytest.fixture(scope="module")
def tiny_gpt(tiny_model_config) -> GPT:
    model = GPT(tiny_model_config)
    model.eval()
    return model


@pytest.fixture
def greedy_gen_config() -> GenerationConfig:
    """Deterministic, no-sampling generation config for fast tests."""
    return GenerationConfig(
        max_length=5,
        do_sample=False,
        use_kv_cache=False,
        use_optimized_sampler=False,
        temperature=1.0,
        pad_token_id=0,
    )


# ---------------------------------------------------------------------------
# Batch fixtures  (pre-built tensors, no tokenizer needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def pretrain_batch() -> dict:
    """Minimal language-model batch: input_ids + labels (shifted)."""
    ids = torch.randint(1, 999, (2, 16))
    labels = ids.clone()
    labels[:, :-1] = ids[:, 1:]
    labels[:, -1] = -100
    return {
        "input_ids": ids,
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "labels": labels,
    }


@pytest.fixture
def classification_batch() -> dict:
    """Minimal classification batch: input_ids + integer class labels."""
    return {
        "input_ids": torch.randint(1, 999, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
    }


@pytest.fixture
def sft_batch() -> dict:
    """Minimal SFT batch with response-only labels."""
    ids = torch.randint(1, 999, (2, 16))
    labels = ids.clone()
    labels[:, :8] = -100   # mask instruction tokens
    return {
        "input_ids": ids,
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# DataLoader utility  (no tokenizer needed — pure tensor batches)
# ---------------------------------------------------------------------------

def make_pretrain_loader(num_samples: int = 6, seq_len: int = 16,
                         vocab_size: int = 1000, batch_size: int = 2) -> DataLoader:
    """
    Build a DataLoader that yields proper tensor-based pretrain batches.
    Avoids relying on ToyDataset's default (tokenizer-less) collation
    which returns Python lists instead of tensors.
    """
    ids = torch.randint(1, vocab_size - 1, (num_samples, seq_len))
    labels = ids.clone()
    labels[:, :-1] = ids[:, 1:]
    labels[:, -1] = -100
    mask = torch.ones(num_samples, seq_len, dtype=torch.long)
    ds = TensorDataset(ids, mask, labels)

    def _collate(items):
        ids_, mask_, lbl_ = zip(*items)
        return {
            "input_ids": torch.stack(ids_),
            "attention_mask": torch.stack(mask_),
            "labels": torch.stack(lbl_),
        }

    return DataLoader(ds, batch_size=batch_size, collate_fn=_collate)


# ---------------------------------------------------------------------------
# Trainer config factory  (avoids WandB + writes to tmp_path)
# ---------------------------------------------------------------------------

def make_trainer_config(config_cls, tmp_path, **overrides):
    """Helper to build a trainer config that skips WandB and uses tmp dir."""
    defaults = dict(
        output_dir=str(tmp_path / "output"),
        num_epochs=1,
        batch_size=2,
        report_to=[],
        logging_steps=1,
        eval_steps=9999,
        save_steps=9999,
        seed=42,
    )
    defaults.update(overrides)
    return config_cls(**defaults)
 
