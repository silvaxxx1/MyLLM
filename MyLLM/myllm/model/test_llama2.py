import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

# Import the llama2 model and other necessary components from llama2_model.py
from llama2 import llama2, pre_compute_freq, apply_rope, FeedForward, RopeFlashAttention, Transformer, LLAMA2_CONFIG_7B


@pytest.fixture
def test_config():
    """Fixture to provide a small test config."""
    return {
        "vocab_size": 32000,
        "emb_dim": 512,
        "num_heads": 8,
        "n_layers": 4,
        "hidden_dim": 2048,
        "context_length": 512,
        "dtype": torch.float32
    }


def test_pre_compute_freq(test_config):
    """Test pre_compute_freq function."""
    freq_tensor = pre_compute_freq(test_config)
    
    # Check tensor shape (context_length, head_dim // 2)
    assert freq_tensor.shape == (test_config["context_length"], test_config["emb_dim"] // 2)
    assert freq_tensor.dtype == torch.complex64
    assert torch.allclose(freq_tensor.abs(), torch.ones_like(freq_tensor.abs(), dtype=torch.float32), atol=1e-4)


def test_apply_rope(test_config):
    """Test apply_rope function."""
    x = torch.randn(2, 4, 8, test_config["emb_dim"] // test_config["num_heads"])
    freqs_complex = pre_compute_freq(test_config)
    output = apply_rope(x, freqs_complex)
    
    # Check that the output has the same shape as input
    assert output.shape == x.shape
    assert not torch.equal(output, x)  # Ensure values have changed


def test_feedforward_shape(test_config):
    """Test that FeedForward does not change the input shape."""
    ff = FeedForward(test_config)
    x = torch.randn(2, 4, test_config["emb_dim"])
    output = ff(x)
    assert output.shape == x.shape


def test_feedforward_forward(test_config):
    """Test FeedForward's forward pass."""
    ff = FeedForward(test_config)
    x = torch.randn(2, 4, test_config["emb_dim"])
    
    # Use patch to mock layers
    with patch.object(ff, 'fc1', MagicMock(return_value=torch.randn(2, 4, test_config["hidden_dim"]))), \
         patch.object(ff, 'fc2', MagicMock(return_value=torch.randn(2, 4, test_config["emb_dim"]))), \
         patch.object(ff, 'fc3', MagicMock(return_value=torch.randn(2, 4, test_config["emb_dim"]))):
        
        output = ff(x)

        # Check that the layers were called
        ff.fc1.assert_called_once()
        ff.fc2.assert_called_once()
        ff.fc3.assert_called_once()


def test_llama2_forward(test_config):
    """Test the forward pass of the Llama2 model."""
    batch_size = 2
    seq_len = 4

    x = torch.randint(0, test_config["vocab_size"], (batch_size, seq_len))
    model = llama2(test_config)
    
    output = model(x)

    # Check that the output shape is (batch_size, seq_len, vocab_size)
    assert output.shape == (batch_size, seq_len, test_config["vocab_size"])
    assert output.dtype == test_config["dtype"]
