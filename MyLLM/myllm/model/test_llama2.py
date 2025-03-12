import pytest
import torch
from llama2 import llama2, pre_compute_freq, apply_rope, FeedForward, RopeFlashAttention, Transformer, RMSNorm


@pytest.fixture
def test_config():
    return {
        "vocab_size": 32000,
        "emb_dim": 512,
        "num_heads": 8,
        "n_layers": 4,
        "hidden_dim": 2048,
        "context_length": 512,
        "dtype": torch.float32,
        "dropout": 0.1  # Add dropout key here
    }


def test_pre_compute_freq(test_config):
    freq_tensor = pre_compute_freq(test_config)
    head_dim = test_config["emb_dim"] // test_config["num_heads"]
    expected_shape = (test_config["context_length"], head_dim // 2)
    
    assert freq_tensor.shape == expected_shape
    assert freq_tensor.dtype == torch.complex64
    assert torch.all(freq_tensor.abs() > 0)


def test_apply_rope(test_config):
    head_dim = test_config["emb_dim"] // test_config["num_heads"]
    batch_size = 2
    num_heads = test_config["num_heads"]
    seq_len = 4
    
    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    freqs_complex = pre_compute_freq(test_config)
    
    output = apply_rope(x, freqs_complex)
    
    assert output.shape == x.shape
    assert not torch.equal(output, x)


def test_rmsnorm():
    dim = 512
    batch_size = 2
    seq_len = 4
    norm = RMSNorm(dim)
    x = torch.randn(batch_size, seq_len, dim)
    output = norm(x)
    
    assert output.shape == x.shape
    std = torch.std(output, dim=-1)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-1)


def test_feedforward_shape(test_config):
    ff = FeedForward(test_config)
    x = torch.randn(2, 4, test_config["emb_dim"])
    output = ff(x)
    
    assert output.shape == x.shape


def test_llama2_forward(test_config):
    model = llama2(test_config)
    x = torch.randint(0, test_config["vocab_size"], (2, 4))
    logits = model(x)
    
    assert logits.shape == (2, 4, test_config["vocab_size"])
