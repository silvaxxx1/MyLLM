"""Tests for GPT model components: MLP, KVCache, Attention, RoPE, full model."""
import pytest
import torch
import warnings

warnings.filterwarnings("ignore", message="CUDA initialization")

from myllm.model import (
    GPT,
    Block,
    CausalSelfAttention,
    GptMLP,
    LLaMAMLP,
    KVCache,
    RMSNorm,
    apply_rope,
    pre_compute_freq,
)
from myllm.Configs import ModelConfig


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt_config():
    """GPT-style config (GptMLP, no RoPE)."""
    return ModelConfig(
        block_size=32, vocab_size=1000, n_layer=2,
        n_head=2, n_embd=64, mlp_class_name="GptMLP",
    )


@pytest.fixture(scope="module")
def llama_config():
    """LLaMA-style config (LLaMAMLP, RoPE, GQA, RMSNorm)."""
    return ModelConfig(
        block_size=32, vocab_size=1000, n_layer=2,
        n_head=4, n_embd=64, n_query_groups=2,
        head_size=16, mlp_hidden_size=128,
        mlp_class_name="LLaMAMLP",
        norm_class_name="RMSNorm",
        use_rope=True, rotary_percentage=1.0,
        parallel_residual=True, norm_eps=1e-5,
    )


@pytest.fixture(autouse=True)
def _clean_cuda():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# MLP components
# ---------------------------------------------------------------------------

class TestGptMLP:

    def test_init(self, gpt_config):
        mlp = GptMLP(gpt_config)
        assert hasattr(mlp, "c_fc") or hasattr(mlp, "fc")  # name may vary

    def test_forward_shape(self, gpt_config):
        mlp = GptMLP(gpt_config)
        x = torch.randn(2, 8, gpt_config.n_embd)
        out = mlp(x)
        assert out.shape == (2, 8, gpt_config.n_embd)
        assert torch.isfinite(out).all()


class TestLLaMAMLP:

    def test_init(self, llama_config):
        mlp = LLaMAMLP(llama_config)
        assert isinstance(mlp.fc_1, torch.nn.Linear)
        assert isinstance(mlp.fc_2, torch.nn.Linear)
        assert isinstance(mlp.proj, torch.nn.Linear)

    def test_weight_shapes(self, llama_config):
        mlp = LLaMAMLP(llama_config)
        assert mlp.fc_1.weight.shape == (llama_config.mlp_hidden_size, llama_config.n_embd)
        assert mlp.proj.weight.shape == (llama_config.n_embd, llama_config.mlp_hidden_size)

    def test_forward_shape(self, llama_config):
        mlp = LLaMAMLP(llama_config)
        x = torch.randn(2, 8, llama_config.n_embd)
        out = mlp(x)
        assert out.shape == (2, 8, llama_config.n_embd)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------

class TestKVCache:

    def _make_cache(self):
        return KVCache(
            batch_size=2, max_seq_len=32,
            num_kv_heads=4, head_dim=16,
            dtype=torch.float32,
        )

    def test_init(self):
        cache = self._make_cache()
        assert cache.size == 0
        assert cache.k_cache.shape == (2, 4, 32, 16)
        assert cache.v_cache.shape == (2, 4, 32, 16)

    def test_update_stores_values(self):
        cache = self._make_cache()
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        k_out, v_out = cache.update(k, v)
        assert cache.size == 8
        assert torch.equal(k_out[:, :, :8], k)
        assert torch.equal(v_out[:, :, :8], v)

    def test_reset_clears_cache(self):
        cache = self._make_cache()
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        cache.update(k, v)
        cache.reset()
        assert cache.size == 0
        assert torch.all(cache.k_cache == 0)
        assert torch.all(cache.v_cache == 0)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:

    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_output_is_finite(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size,seq_len", [(2, 16), (1, 32), (4, 8)])
def test_attention_output_shape(gpt_config, batch_size, seq_len):
    attn = CausalSelfAttention(gpt_config, block_idx=0)
    x = torch.randn(batch_size, seq_len, gpt_config.n_embd)
    out = attn(x)
    assert out.shape == (batch_size, seq_len, gpt_config.n_embd)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_size,n_head,n_embd", [
    (32, 4, 64),
    (64, 8, 128),
    (16, 2, 32),
])
def test_rope_preserves_shape(block_size, n_head, n_embd):
    cfg = ModelConfig(
        block_size=block_size, n_head=n_head, n_embd=n_embd,
        use_rope=True, vocab_size=1000, n_layer=2,
    )
    freqs = pre_compute_freq(cfg, context_length=block_size)
    x = torch.randn(2, n_head, block_size, n_embd // n_head)
    rotated = apply_rope(x, freqs)
    assert rotated.shape == x.shape
    assert not torch.equal(rotated, x)
    assert not torch.isnan(rotated).any()


# ---------------------------------------------------------------------------
# Full GPT model
# ---------------------------------------------------------------------------

class TestGPTModel:

    def test_init(self, gpt_config):
        model = GPT(gpt_config)
        assert isinstance(model.lm_head, torch.nn.Linear)
        assert isinstance(model.wte, torch.nn.Embedding)
        assert len(model.transformer) == gpt_config.n_layer

    def test_forward_shape(self, gpt_config):
        model = GPT(gpt_config).eval()
        x = torch.randint(0, gpt_config.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(x)
        # output vocab dim may be padded
        assert out.shape[:2] == (2, 16)
        assert out.shape[2] >= gpt_config.vocab_size

    def test_output_is_finite(self, gpt_config):
        model = GPT(gpt_config).eval()
        x = torch.randint(0, gpt_config.vocab_size, (1, 8))
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_single_token_forward(self, gpt_config):
        model = GPT(gpt_config).eval()
        x = torch.randint(0, gpt_config.vocab_size, (1, 1))
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 1

    def test_kv_cache_autoregressive(self, gpt_config):
        model = GPT(gpt_config).eval()
        model.initialize_kv_cache(batch_size=1, max_seq_len=32)
        x = torch.randint(0, gpt_config.vocab_size, (1, 1))
        with torch.no_grad():
            for _ in range(5):
                out = model(x, use_cache=True)
                assert out.shape[0] == 1
                x = out[:, -1:].argmax(dim=-1)
        model.reset_cache()
        assert not model.kv_cache_initialized

    def test_llama_config_forward(self, llama_config):
        model = GPT(llama_config).eval()
        x = torch.randint(0, llama_config.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(x)
        assert out.shape[:2] == (2, 16)
        assert torch.isfinite(out).all()

    def test_parameter_count_positive(self, gpt_config):
        model = GPT(gpt_config)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0
