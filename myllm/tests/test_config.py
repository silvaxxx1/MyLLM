import sys 
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import json
from config import Config

def test_config_init_defaults():
    cfg = Config()
    assert cfg.block_size == 1024
    assert cfg.vocab_size == 50257
    assert cfg.n_layer == 12
    assert cfg.head_size == cfg.n_embd // cfg.n_head

def test_config_from_name():
    cfg = Config.from_name("gpt2-small")
    assert cfg.name == "gpt2-small"
    assert cfg.n_layer == 12

def test_config_save_and_load():
    cfg = Config.from_name("gpt2-small")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    cfg.save(path)
    cfg_loaded = Config.load(path)
    assert cfg_loaded.name == cfg.name
    os.remove(path)

def test_config_update():
    cfg = Config()
    cfg.update(block_size=2048, learning_rate=1e-4)
    assert cfg.block_size == 2048
    assert cfg.learning_rate == 1e-4

def test_invalid_update():
    cfg = Config()
    cfg.update(does_not_exist=42)  # Should not raise error, just skip

def test_validation_passes():
    cfg = Config(n_embd=768, n_head=12)
    cfg.validate()

def test_validation_fails():
    with pytest.raises(AssertionError):
        Config(n_embd=770, n_head=12).validate()  # 770 not divisible by 12

def test_estimate_memory():
    cfg = Config()
    estimate = cfg.estimate_memory(batch_size=2)
    assert "total_gb" in estimate
    assert estimate["n_parameters"] > 0

def test_get_trainable_params():
    cfg = Config()
    params = cfg.get_trainable_params()
    assert isinstance(params, dict)
    assert "n_layer" in params
