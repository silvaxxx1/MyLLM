"""Tests for ModelConfig and GenerationConfig."""
import os
import pytest
import tempfile
from myllm.Configs import ModelConfig, GenerationConfig


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfigPresets:

    def test_gpt2_small_loads(self):
        cfg = ModelConfig.from_name("gpt2-small")
        assert cfg.n_layer == 12
        assert cfg.n_embd > 0
        assert cfg.vocab_size > 0

    def test_gpt2_medium_is_larger(self):
        small = ModelConfig.from_name("gpt2-small")
        medium = ModelConfig.from_name("gpt2-medium")
        assert medium.n_embd >= small.n_embd

    def test_unknown_preset_raises(self):
        with pytest.raises(Exception):
            ModelConfig.from_name("not-a-real-model-xyz")


class TestModelConfigCustom:

    def test_fields_set_correctly(self, tiny_model_config):
        assert tiny_model_config.n_layer == 2
        assert tiny_model_config.n_embd == 64
        assert tiny_model_config.n_head == 2
        assert tiny_model_config.vocab_size == 1000
        assert tiny_model_config.block_size == 32

    def test_n_embd_not_divisible_by_n_head_raises(self):
        with pytest.raises(Exception):
            ModelConfig(
                n_layer=2, n_embd=65, n_head=2,
                vocab_size=1000, block_size=32,
                mlp_class_name="GptMLP",
            )

    def test_update_fields(self):
        cfg = ModelConfig.from_name("gpt2-small")
        cfg.update(block_size=512)
        assert cfg.block_size == 512

    def test_estimate_memory_returns_dict(self):
        cfg = ModelConfig.from_name("gpt2-small")
        mem = cfg.estimate_memory(batch_size=1)
        assert isinstance(mem, dict)
        assert "n_parameters" in mem
        assert mem["n_parameters"] > 0

    def test_get_trainable_params(self):
        cfg = ModelConfig.from_name("gpt2-small")
        params = cfg.get_trainable_params()
        assert isinstance(params, dict)
        assert "n_layer" in params

    def test_save_and_load(self):
        cfg = ModelConfig.from_name("gpt2-small")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            loaded = ModelConfig.load(path)
            assert loaded.n_layer == cfg.n_layer
            assert loaded.n_embd == cfg.n_embd
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------------------

class TestGenerationConfig:

    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.max_length == 20
        assert cfg.temperature == 1.0
        assert cfg.do_sample is True
        assert cfg.use_kv_cache is True
        assert cfg.apply_repetition_penalty is True
        assert cfg.apply_top_k_sampling is True
        assert cfg.return_tokens is True
        assert cfg.use_optimized_sampler is True

    def test_custom_values(self):
        cfg = GenerationConfig(
            max_length=50,
            temperature=0.7,
            top_k=10,
            repetition_penalty=1.2,
            eos_token_ids=[101],
            return_logprobs=True,
            batch_size=4,
        )
        assert cfg.max_length == 50
        assert cfg.temperature == 0.7
        assert cfg.top_k == 10
        assert cfg.repetition_penalty == 1.2
        assert cfg.eos_token_ids == [101]
        assert cfg.return_logprobs is True
        assert cfg.batch_size == 4

    def test_greedy_decoding(self):
        cfg = GenerationConfig(do_sample=False, use_kv_cache=False)
        assert cfg.do_sample is False
        assert cfg.use_kv_cache is False
