"""Tests for the LLM inference wrapper (myllm/api.py)."""
import pytest
import torch
from myllm.api import LLM
from myllm.Configs import ModelConfig, GenerationConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def llm(tiny_model_config):
    model = LLM(config=tiny_model_config, device="cpu")
    model.model.eval()
    return model


class DummyTokenizer:
    """Minimal tokenizer stub — no real weights needed."""
    pad_token_id = 0
    vocab_size = 1000

    def encode(self, text, return_tensors=None):
        ids = [hash(w) % 998 + 1 for w in text.split()]
        if not ids:
            ids = [1]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, tokens, skip_special_tokens=True):
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return " ".join(str(t) for t in tokens if t != self.pad_token_id)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestLLMInit:

    def test_model_is_not_none(self, llm):
        assert llm.model is not None

    def test_model_is_nn_module(self, llm):
        import torch.nn as nn
        assert isinstance(llm.model, nn.Module)

    def test_device_set(self, llm):
        assert llm.device == "cpu"

    def test_init_without_config_is_empty(self):
        llm = LLM(device="cpu")
        assert llm.model is None


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

class TestGenerate:

    def test_returns_dict_with_tokens(self, llm, greedy_gen_config):
        x = torch.randint(1, 999, (1, 5))
        out = llm.generate(x, greedy_gen_config)
        assert isinstance(out, dict)
        assert "tokens" in out

    def test_output_longer_than_input(self, llm, greedy_gen_config):
        x = torch.randint(1, 999, (1, 5))
        out = llm.generate(x, greedy_gen_config)
        assert out["tokens"].shape[1] > 5

    def test_greedy_is_deterministic(self, llm, greedy_gen_config):
        x = torch.randint(1, 999, (1, 5))
        out1 = llm.generate(x, greedy_gen_config)
        out2 = llm.generate(x, greedy_gen_config)
        assert torch.equal(out1["tokens"], out2["tokens"])

    def test_batch_generation(self, llm, greedy_gen_config):
        x = torch.randint(1, 999, (3, 5))
        out = llm.generate(x, greedy_gen_config)
        assert out["tokens"].shape[0] == 3

    def test_with_kv_cache(self, llm, tiny_model_config):
        cfg = GenerationConfig(
            max_length=5, do_sample=False, use_kv_cache=True,
            use_optimized_sampler=False, apply_repetition_penalty=False,
            apply_top_k_sampling=False, apply_top_p_sampling=False,
            temperature=1.0, pad_token_id=0,
        )
        x = torch.randint(1, tiny_model_config.vocab_size, (1, 4))
        out = llm.generate(x, cfg)
        assert out["tokens"].shape[1] > 4

    def test_top_k_sampling(self, llm):
        cfg = GenerationConfig(
            max_length=5, do_sample=True, use_kv_cache=False,
            use_optimized_sampler=True, apply_top_k_sampling=True,
            top_k=10, apply_top_p_sampling=False,
            apply_repetition_penalty=False, temperature=1.0, pad_token_id=0,
        )
        x = torch.randint(1, 999, (1, 5))
        out = llm.generate(x, cfg)
        assert "tokens" in out

    def test_top_p_sampling(self, llm):
        cfg = GenerationConfig(
            max_length=5, do_sample=True, use_kv_cache=False,
            use_optimized_sampler=True, apply_top_k_sampling=False,
            apply_top_p_sampling=True, top_p=0.9,
            apply_repetition_penalty=False, temperature=1.0, pad_token_id=0,
        )
        x = torch.randint(1, 999, (1, 5))
        out = llm.generate(x, cfg)
        assert "tokens" in out

    def test_logprobs_returned_when_requested(self, llm, tiny_model_config):
        cfg = GenerationConfig(
            max_length=5, do_sample=False, use_kv_cache=False,
            use_optimized_sampler=False, apply_repetition_penalty=False,
            apply_top_k_sampling=False, apply_top_p_sampling=False,
            temperature=1.0, pad_token_id=0, return_logprobs=True,
        )
        x = torch.randint(1, tiny_model_config.vocab_size, (1, 5))
        out = llm.generate(x, cfg)
        assert "logprobs" in out
        assert out["logprobs"] is not None

    def test_output_tokens_are_integers(self, llm, greedy_gen_config):
        x = torch.randint(1, 999, (1, 5))
        out = llm.generate(x, greedy_gen_config)
        assert out["tokens"].dtype == torch.long


# ---------------------------------------------------------------------------
# generate_text()
# ---------------------------------------------------------------------------

class TestGenerateText:

    def test_returns_dict_with_text(self, llm, greedy_gen_config):
        tok = DummyTokenizer()
        result = llm.generate_text("hello world", tok, greedy_gen_config)
        assert isinstance(result, dict)
        assert "text" in result
        assert isinstance(result["text"], str)

    def test_returns_tokens_when_requested(self, llm, greedy_gen_config):
        tok = DummyTokenizer()
        greedy_gen_config.return_tokens = True
        result = llm.generate_text("hello world", tok, greedy_gen_config)
        assert result["tokens"] is not None

    def test_empty_prompt_does_not_crash(self, llm, greedy_gen_config):
        tok = DummyTokenizer()
        result = llm.generate_text("", tok, greedy_gen_config)
        assert "text" in result


# ---------------------------------------------------------------------------
# generate_batch()
# ---------------------------------------------------------------------------

class TestGenerateBatch:

    def test_returns_list_of_dicts(self, llm, greedy_gen_config):
        tok = DummyTokenizer()
        prompts = ["hello world", "foo bar baz"]
        results = llm.generate_batch(prompts, tok, greedy_gen_config)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert "text" in r

    def test_empty_prompts_returns_empty(self, llm, greedy_gen_config):
        tok = DummyTokenizer()
        results = llm.generate_batch([], tok, greedy_gen_config)
        assert results == []

    def test_single_prompt_batch(self, llm, greedy_gen_config):
        tok = DummyTokenizer()
        results = llm.generate_batch(["hello"], tok, greedy_gen_config)
        assert len(results) == 1
