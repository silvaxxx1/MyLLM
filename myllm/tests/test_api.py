import sys
import os

# Add the parent directory to the sys.path so we can import api, config, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
warnings.filterwarnings("ignore", message="CUDA initialization: CUDA unknown error")


import torch
import pytest
from api import LLM, OptimizedSampler
from Configs.ModelConfig import ModelConfig
from Configs.GenConfig import GenerationConfig

# Dummy tokenizer mock
class DummyTokenizer:
    def __init__(self):
        self.vocab = {f"token_{i}": i for i in range(100)}
        self.pad_token_id = 0

    def encode(self, text, return_tensors="pt"):
        tokens = [self.vocab.get(tok, 1) for tok in text.split()]
        return torch.tensor([tokens], dtype=torch.long)

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join([f"token_{i}" for i in tokens if i != self.pad_token_id])


# Dummy config for small GPT (stub only)
@pytest.fixture
def dummy_model_config():
    config = ModelConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=16
    )
    config.mlp_class_name = "GptMLP"  # <-- Use the correct MLP class to avoid attribute errors
    config.__post_init__()  # Recalculate derived attributes after manual change
    return config

@pytest.fixture
def dummy_generation_config():
    return GenerationConfig(
        max_length=5,
        temperature=1.0,
        repetition_penalty=1.1,
        return_logprobs=True,
        return_tokens=True,
        use_kv_cache=False,
        use_optimized_sampler=True,
        apply_repetition_penalty=True,
        apply_top_k_sampling=True,
        top_k=5,
        eos_token_ids=[99]
    )


# ========== OptimizedSampler Tests ==========

def test_repetition_penalty_vectorized():
    logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0]
    ])
    tokens = torch.tensor([
        [0, 1],
        [1, 2]
    ])
    penalty = 2.0
    out = OptimizedSampler.apply_repetition_penalty_vectorized(logits, tokens, penalty)
    
    assert out[0, 0] == pytest.approx(1.0 / penalty)
    assert out[1, 2] == pytest.approx(3.0 / penalty)
    assert out.shape == logits.shape


def test_combined_top_k_top_p_sampling():
    logits = torch.tensor([
        [0.1, 0.3, 0.2, 0.4],
        [0.5, 0.2, 0.1, 0.2]
    ])
    top_k = 2
    top_p = 0.9
    filtered = OptimizedSampler.combined_top_k_top_p_sampling(logits, top_k=top_k, top_p=top_p)

    # Only top 2 logits should be non-inf
    assert torch.isinf(filtered).sum().item() >= 2


def test_check_eos_vectorized_all_match():
    tokens = torch.tensor([99, 99, 99])
    assert OptimizedSampler.check_eos_vectorized(tokens, eos_token_ids=[99]) is True


def test_check_eos_vectorized_partial_match():
    tokens = torch.tensor([99, 10, 99])
    assert OptimizedSampler.check_eos_vectorized(tokens, eos_token_ids=[99]) is False


# ========== LLM Generation Tests ==========

def test_llm_generation(dummy_model_config, dummy_generation_config):
    tokenizer = DummyTokenizer()
    llm = LLM(config=dummy_model_config, device="cpu")
    
    prompt = "token_1 token_2"
    result = llm.generate_text(prompt, tokenizer, dummy_generation_config)

    assert isinstance(result, dict)
    assert "text" in result
    assert "tokens" in result
    assert isinstance(result["tokens"], list)
    assert isinstance(result["text"], str)
    assert len(result["tokens"]) >= len(tokenizer.encode(prompt)[0])

def test_llm_batch_generation(dummy_model_config, dummy_generation_config):
    tokenizer = DummyTokenizer()
    llm = LLM(config=dummy_model_config, device="cpu")

    prompts = ["token_1 token_2", "token_3 token_4"]
    result = llm.generate_batch(prompts, tokenizer, dummy_generation_config)

    assert isinstance(result, list)
    assert len(result) == 2
    for r in result:
        assert "text" in r and isinstance(r["text"], str)
        assert "tokens" in r and isinstance(r["tokens"], list)
