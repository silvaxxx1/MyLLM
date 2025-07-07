import pytest
import torch
import math
import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (
    GPT,
    Block,
    CausalSelfAttention,
    GptMLP,
    LLaMAMLP,
    KVCache,
    RMSNorm,
    apply_rope,
    pre_compute_freq
)
from config import Config

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

@pytest.fixture
def test_config():
    """Create a minimal test configuration"""
    return Config(
        block_size=32,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=64,
        n_query_groups=2,
        head_size=16,
        mlp_hidden_size=128,
        dropout=0.1,
        bias=False,
        padded_vocab_size=1024,
        attention_bias=False,
        mlp_class_name="LLaMAMLP",
        norm_class_name="RMSNorm",
        use_rope=True,
        rotary_percentage=1.0,
        parallel_residual=True,
        norm_eps=1e-5
    )

@pytest.fixture
def device():
    """Get the default device for testing"""
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda:0")
    except RuntimeError:
        pass
    return torch.device("cpu")

class TestLLaMAMLP:
    def test_initialization(self, test_config):
        mlp = LLaMAMLP(test_config)
        assert isinstance(mlp.fc_1, torch.nn.Linear)
        assert isinstance(mlp.fc_2, torch.nn.Linear)
        assert isinstance(mlp.proj, torch.nn.Linear)
        
        # Test shapes
        assert mlp.fc_1.weight.shape == (test_config.mlp_hidden_size, test_config.n_embd)
        assert mlp.fc_2.weight.shape == (test_config.mlp_hidden_size, test_config.n_embd)
        assert mlp.proj.weight.shape == (test_config.n_embd, test_config.mlp_hidden_size)
        
    def test_forward(self, test_config, device):
        mlp = LLaMAMLP(test_config).to(device)
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, test_config.n_embd).to(device)
        
        output = mlp(x)
        assert output.shape == (batch_size, seq_len, test_config.n_embd)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestKVCache:
    def test_initialization(self):
        cache = KVCache(
            batch_size=2,
            max_seq_len=32,
            num_kv_heads=4,
            head_dim=16,
            dtype=torch.float32
        )
        assert cache.size == 0
        assert cache.k_cache.shape == (2, 4, 32, 16)
        assert cache.v_cache.shape == (2, 4, 32, 16)
    
    def test_update(self, device):
        cache = KVCache(
            batch_size=2,
            max_seq_len=32,
            num_kv_heads=4,
            head_dim=16,
            dtype=torch.float32
        ).to(device)
        
        k_val = torch.randn(2, 4, 8, 16).to(device)
        v_val = torch.randn(2, 4, 8, 16).to(device)
        
        k_out, v_out = cache.update(k_val, v_val)
        assert cache.size == 8
        assert torch.equal(k_out[:, :, :8], k_val)
        assert torch.equal(v_out[:, :, :8], v_val)
        
    def test_reset(self, device):
        cache = KVCache(
            batch_size=2,
            max_seq_len=32,
            num_kv_heads=4,
            head_dim=16,
            dtype=torch.float32
        ).to(device)
        
        k_val = torch.randn(2, 4, 8, 16).to(device)
        v_val = torch.randn(2, 4, 8, 16).to(device)
        cache.update(k_val, v_val)
        
        cache.reset()
        assert cache.size == 0
        assert torch.all(cache.k_cache == 0)
        assert torch.all(cache.v_cache == 0)

class TestGPT:
    def test_initialization(self, test_config):
        model = GPT(test_config)
        assert isinstance(model.lm_head, torch.nn.Linear)
        assert isinstance(model.wte, torch.nn.Embedding)
        assert len(model.transformer) == test_config.n_layer
        
    def test_forward(self, test_config, device):
        model = GPT(test_config).to(device)
        batch_size, seq_len = 2, 16
        x = torch.randint(0, test_config.vocab_size, (batch_size, seq_len)).to(device)
        
        output = model(x)
        expected_shape = (batch_size, seq_len, test_config.padded_vocab_size)
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()
        
    def test_kv_cache(self, test_config, device):
        model = GPT(test_config).to(device)
        model.initialize_kv_cache(batch_size=1, max_seq_len=32)
        
        x = torch.randint(0, test_config.vocab_size, (1, 1)).to(device)
        
        with torch.no_grad():
            for _ in range(5):
                output = model(x, use_cache=True)
                assert output.shape == (1, 1, test_config.padded_vocab_size)
                x = output[:, -1:].argmax(dim=-1)
        
        model.reset_cache()
        assert not model.kv_cache_initialized
        
    def test_memory_cleanup(self, test_config, device):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
            
            model = GPT(test_config).to(device)
            x = torch.randint(0, test_config.vocab_size, (1, 32)).to(device)
            
            with torch.no_grad():
                output = model(x)
                del output
            
            model.cpu()
            del model
            torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated()
            allowed_margin = 1_000_000  # Allow 1MB fluctuation
            assert final_memory <= initial_memory + allowed_margin, (
                f"Memory leak detected: initial={initial_memory}, final={final_memory}"
            )


@pytest.mark.parametrize("batch_size,seq_len", [(2, 16), (1, 32), (4, 8)])
def test_attention_shapes(test_config, device, batch_size, seq_len):
    attn = CausalSelfAttention(test_config, block_idx=0).to(device)
    x = torch.randn(batch_size, seq_len, test_config.n_embd).to(device)
    
    output = attn(x)
    assert output.shape == (batch_size, seq_len, test_config.n_embd)
    assert not torch.isnan(output).any()

@pytest.mark.parametrize("block_size,n_head,n_embd", [
    (32, 4, 64),
    (64, 8, 128),
    (16, 2, 32)
])
def test_rope(block_size, n_head, n_embd):
    config = Config(
        block_size=block_size,
        n_head=n_head,
        n_embd=n_embd,
        use_rope=True
    )
    
    freqs = pre_compute_freq(config, context_length=block_size)
    x = torch.randn(2, n_head, block_size, n_embd // n_head)
    
    rotated = apply_rope(x, freqs)
    assert rotated.shape == x.shape
    assert not torch.equal(rotated, x)
    assert not torch.isnan(rotated).any()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])