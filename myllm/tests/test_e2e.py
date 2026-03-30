"""
End-to-end tests: init → train → checkpoint → generate.

These tests exercise the full pipeline with a tiny randomly-initialised
model — no pretrained weights required.
"""
import os
import pytest
import torch

from myllm.Configs import ModelConfig, GenerationConfig
from myllm.model import GPT
from myllm.api import LLM
from myllm.Train.configs.TrainerConfig import TrainerConfig
from myllm.Train.trainer import PretrainTrainer
from myllm.tests.conftest import make_trainer_config, make_pretrain_loader


TINY = ModelConfig(
    n_layer=2, n_embd=64, n_head=2,
    vocab_size=1000, block_size=32,
    mlp_class_name="GptMLP",
)

GREEDY = GenerationConfig(
    max_length=8, do_sample=False, use_kv_cache=False,
    use_optimized_sampler=False, apply_repetition_penalty=False,
    apply_top_k_sampling=False, apply_top_p_sampling=False,
    temperature=1.0, pad_token_id=0,
)


# ---------------------------------------------------------------------------
# Stage 1 – Model initialisation
# ---------------------------------------------------------------------------

def test_model_init_and_forward():
    """GPT with tiny config produces correct output shape."""
    model = GPT(TINY).eval()
    x = torch.randint(1, 999, (1, 8))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape[:2] == (1, 8)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Stage 2 – Training loop
# ---------------------------------------------------------------------------

def test_multiple_training_steps_produce_valid_losses(tmp_path):
    """Three training steps all return positive finite losses."""
    cfg = make_trainer_config(TrainerConfig, tmp_path)
    trainer = PretrainTrainer(cfg, model_config=TINY)
    trainer.setup_model()
    trainer.setup_optimizer()

    dl = make_pretrain_loader(num_samples=6, seq_len=16, vocab_size=1000, batch_size=2)
    losses = []
    for batch in dl:
        batch = trainer._prepare_batch(batch)
        result = trainer.training_step(batch)
        losses.append(result["loss"])

    assert len(losses) == 3
    assert all(l > 0 for l in losses)
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


def test_loss_decreases_or_stays_finite_over_steps(tmp_path):
    """Loss should never go nan/inf during training steps."""
    cfg = make_trainer_config(TrainerConfig, tmp_path)
    trainer = PretrainTrainer(cfg, model_config=TINY)
    trainer.setup_model()
    trainer.setup_optimizer()

    dl = make_pretrain_loader(num_samples=10, seq_len=16, vocab_size=1000, batch_size=2)
    for batch in dl:
        batch = trainer._prepare_batch(batch)
        result = trainer.training_step(batch)
        assert torch.isfinite(torch.tensor(result["loss"]))


# ---------------------------------------------------------------------------
# Stage 3 – Checkpointing
# ---------------------------------------------------------------------------

def test_checkpoint_saved_to_disk(tmp_path):
    """save_checkpoint writes a file or directory to disk."""
    cfg = make_trainer_config(TrainerConfig, tmp_path)
    trainer = PretrainTrainer(cfg, model_config=TINY)
    trainer.setup_model()

    ckpt = trainer.save_checkpoint(str(tmp_path / "checkpoint"))
    assert os.path.exists(ckpt) or os.path.isdir(ckpt)


def test_checkpoint_load_restores_weights(tmp_path):
    """Weights loaded from checkpoint match saved weights."""
    cfg = make_trainer_config(TrainerConfig, tmp_path)
    trainer = PretrainTrainer(cfg, model_config=TINY)
    trainer.setup_model()

    # Record a weight before saving
    param_before = next(trainer.model.parameters()).detach().clone()

    ckpt = trainer.save_checkpoint(str(tmp_path / "checkpoint"))

    # Corrupt the weight
    with torch.no_grad():
        next(trainer.model.parameters()).fill_(99.0)

    # Reload
    trainer.load_checkpoint(ckpt)

    param_after = next(trainer.model.parameters()).detach()
    assert torch.allclose(param_before, param_after)


# ---------------------------------------------------------------------------
# Stage 4 – Inference
# ---------------------------------------------------------------------------

def test_generate_from_random_model():
    """LLM wrapper generates tokens with a randomly-initialised model."""
    llm = LLM(config=TINY, device="cpu")
    llm.model.eval()

    x = torch.randint(1, 999, (1, 5))
    out = llm.generate(x, GREEDY)
    assert "tokens" in out
    assert out["tokens"].shape[1] > 5


def test_generate_with_kv_cache():
    """KV-cache path produces same token count as non-cached path."""
    llm = LLM(config=TINY, device="cpu")
    llm.model.eval()

    cached_cfg = GenerationConfig(
        max_length=8, do_sample=False, use_kv_cache=True,
        use_optimized_sampler=False, apply_repetition_penalty=False,
        apply_top_k_sampling=False, apply_top_p_sampling=False,
        temperature=1.0, pad_token_id=0,
    )
    x = torch.randint(1, 999, (1, 5))
    out_cached = llm.generate(x, cached_cfg)
    out_plain = llm.generate(x, GREEDY)

    assert out_cached["tokens"].shape == out_plain["tokens"].shape


# ---------------------------------------------------------------------------
# Stage 5 – Full pipeline
# ---------------------------------------------------------------------------

def test_train_then_generate(tmp_path):
    """Train for a few steps then use the model for generation."""
    # Train
    cfg = make_trainer_config(TrainerConfig, tmp_path)
    trainer = PretrainTrainer(cfg, model_config=TINY)
    trainer.setup_model()
    trainer.setup_optimizer()

    dl = make_pretrain_loader(num_samples=4, seq_len=16, vocab_size=1000, batch_size=2)
    for batch in dl:
        batch = trainer._prepare_batch(batch)
        trainer.training_step(batch)

    # Transfer trained weights to LLM wrapper (move to CPU for inference)
    llm = LLM(config=TINY, device="cpu")
    llm.model = trainer.model.cpu()
    llm.model.eval()

    # Generate
    x = torch.randint(1, 999, (1, 5))
    out = llm.generate(x, GREEDY)
    assert "tokens" in out
    assert out["tokens"].shape[1] > 5
    assert torch.isfinite(out["tokens"].float()).all()
