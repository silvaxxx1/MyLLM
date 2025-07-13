import sys 
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import json
from Configs.ModelConfig import ModelConfig
from Configs.GenConfig import GenerationConfig


def test_config_init_defaults():
    cfg = ModelConfig()
    assert cfg.block_size == 1024
    assert cfg.vocab_size == 50257
    assert cfg.n_layer == 12
    assert cfg.head_size == cfg.n_embd // cfg.n_head

def test_config_from_name():
    cfg = ModelConfig.from_name("gpt2-small")
    assert cfg.name == "gpt2-small"
    assert cfg.n_layer == 12

def test_config_save_and_load():
    cfg = ModelConfig.from_name("gpt2-small")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    cfg.save(path)
    cfg_loaded = ModelConfig.load(path)
    assert cfg_loaded.name == cfg.name
    os.remove(path)

def test_config_update():
    cfg = ModelConfig()
    cfg.update(block_size=2048, learning_rate=1e-4)
    assert cfg.block_size == 2048
    assert cfg.learning_rate == 1e-4

def test_invalid_update():
    cfg = ModelConfig()
    cfg.update(does_not_exist=42)  # Should not raise error, just skip

def test_validation_passes():
    cfg = ModelConfig(n_embd=768, n_head=12)
    cfg.validate()

def test_validation_fails():
    with pytest.raises(AssertionError):
        ModelConfig(n_embd=770, n_head=12).validate()  # 770 not divisible by 12

def test_estimate_memory():
    cfg = ModelConfig()
    estimate = cfg.estimate_memory(batch_size=2)
    assert "total_gb" in estimate
    assert estimate["n_parameters"] > 0

def test_get_trainable_params():
    cfg = ModelConfig()
    params = cfg.get_trainable_params()
    assert isinstance(params, dict)
    assert "n_layer" in params 


def test_generation_config_defaults():
    cfg = GenerationConfig()
    assert cfg.max_length == 20
    assert cfg.temperature == 1.0
    assert cfg.do_sample is True
    assert cfg.use_kv_cache is True
    assert cfg.apply_repetition_penalty is True
    assert cfg.apply_top_k_sampling is True
    assert cfg.return_tokens is True
    assert cfg.batch_size == 1
    assert cfg.use_optimized_sampler is True

def test_generation_config_custom_values():
    cfg = GenerationConfig(
        max_length=50,
        temperature=0.7,
        top_k=10,
        repetition_penalty=1.2,
        eos_token_ids=[101],
        return_logprobs=True,
        batch_size=4,
        use_optimized_sampler=False
    )
    assert cfg.max_length == 50
    assert cfg.temperature == 0.7
    assert cfg.top_k == 10
    assert cfg.repetition_penalty == 1.2
    assert cfg.eos_token_ids == [101]
    assert cfg.return_logprobs is True
    assert cfg.batch_size == 4
    assert cfg.use_optimized_sampler is False
