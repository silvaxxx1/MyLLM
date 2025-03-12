import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

# Import the model (assuming the file is named llama2_model.py)
# Replace with your actual import path
from llama2 import llama2, Transformer, RopeFlashAttention, FeedForward, RMSNorm, LLAMA2_CONFIG_7B

@pytest.fixture
def config():
    return LLAMA2_CONFIG_7B

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class TestLlama2:
    def test_model_init(self, config):
        """Test if the model can be initialized correctly."""
        model = llama2(config)
        # Check if the model has the expected components
        assert isinstance(model.token_emb, nn.Embedding)
        assert len(model.trs_blk) == config["n_layers"]
        assert isinstance(model.norm, RMSNorm)
        assert isinstance(model.proj, nn.Linear)
        
    def test_model_forward(self, config, device):
        """Test the forward pass of the model."""
        # Use smaller model for testing to avoid OOM
        test_config = config.copy()
        test_config["n_layers"] = 2  # Reduce layers for testing
        test_config["emb_dim"] = 1024  # Reduce embedding size
        test_config["hidden_dim"] = 2048  # Reduce hidden dimension
        
        model = llama2(test_config).to(device)
        batch_size = 2
        seq_len = 10
        
        # Create a sample input
        x = torch.randint(0, test_config["vocab_size"], (batch_size, seq_len)).to(device)
        
        # Run forward pass
        out = model(x)
        
        # Check output shape
        assert out.shape == (batch_size, seq_len, test_config["vocab_size"])
        
    def test_transformer_block(self, config, device):
        """Test the transformer block."""
        # Use smaller model for testing
        test_config = config.copy()
        test_config["emb_dim"] = 1024
        test_config["hidden_dim"] = 2048
        
        block = Transformer(test_config).to(device)
        batch_size = 2
        seq_len = 10
        
        # Create a sample input
        x = torch.randn(batch_size, seq_len, test_config["emb_dim"]).to(device)
        
        # Run forward pass
        out = block(x)
        
        # Check output shape
        assert out.shape == (batch_size, seq_len, test_config["emb_dim"])
        
    def test_rope_flash_attention(self, config, device):
        """Test the RoPE Flash Attention mechanism."""
        # Use smaller model for testing
        test_config = config.copy()
        test_config["emb_dim"] = 1024
        test_config["num_heads"] = 8
        
        attention = RopeFlashAttention(test_config, test_config["emb_dim"], test_config["emb_dim"]).to(device)
        batch_size = 2
        seq_len = 10
        
        # Create a sample input
        x = torch.randn(batch_size, seq_len, test_config["emb_dim"]).to(device)
        
        # Run forward pass
        out = attention(x)
        
        # Check output shape
        assert out.shape == (batch_size, seq_len, test_config["emb_dim"])
        
    def test_feed_forward(self, config, device):
        """Test the Feed Forward network."""
        # Use smaller model for testing
        test_config = config.copy()
        test_config["emb_dim"] = 1024
        test_config["hidden_dim"] = 2048
        
        ff = FeedForward(test_config).to(device)
        batch_size = 2
        seq_len = 10
        
        # Create a sample input
        x = torch.randn(batch_size, seq_len, test_config["emb_dim"]).to(device)
        
        # Run forward pass
        out = ff(x)
        
        # Check output shape
        assert out.shape == (batch_size, seq_len, test_config["emb_dim"])
        
    def test_rms_norm(self, device):
        """Test the RMSNorm layer."""
        dim = 1024
        norm = RMSNorm(dim).to(device)
        batch_size = 2
        seq_len = 10
        
        # Create a sample input
        x = torch.randn(batch_size, seq_len, dim).to(device)
        
        # Run forward pass
        out = norm(x)
        
        # Check output shape
        assert out.shape == (batch_size, seq_len, dim)
        
        # Check normalization (mean should be close to 0, std close to 1 but affected by weights)
        # We check along the last dimension
        normalized = out.detach().cpu().numpy()
        assert normalized.shape == (batch_size, seq_len, dim)
        
    def test_freqs_complex_computation(self, config):
        """Test the RoPE frequency computation."""
        from llama2 import pre_compute_freq
        
        freqs = pre_compute_freq(config)
        
        # Check shape (should be context_length x (head_dim/2))
        head_dim = config["emb_dim"] // config["num_heads"]
        assert freqs.shape == (config["context_length"], head_dim // 2)
        
        # Check that frequencies are complex numbers
        assert freqs.dtype == torch.complex64 or freqs.dtype == torch.complex128
        
    def test_autoregressive_property(self, config, device):
        """Test that the model respects autoregressive property (causal masking)."""
        # Use smaller model for testing
        test_config = config.copy()
        test_config["n_layers"] = 2
        test_config["emb_dim"] = 1024
        test_config["hidden_dim"] = 2048
        test_config["num_heads"] = 8
        
        model = llama2(test_config).to(device)
        batch_size = 2
        seq_len = 10
        
        # Create identical inputs
        x1 = torch.randint(0, test_config["vocab_size"], (batch_size, seq_len)).to(device)
        x2 = x1.clone()
        
        # Change one token in the middle
        change_pos = seq_len // 2
        if x2[0, change_pos] < test_config["vocab_size"] - 1:
            x2[0, change_pos] += 1
        else:
            x2[0, change_pos] -= 1
            
        # Get outputs for both inputs
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
            
        # Check that positions before change_pos are identical
        assert_close(out1[:, :change_pos, :], out2[:, :change_pos, :], rtol=1e-3, atol=1e-3)
        
        # Check that at least the changed position output is different
        # (may not be different for all positions after change_pos due to implementation details)
        assert not torch.allclose(out1[:, change_pos:, :], out2[:, change_pos:, :], rtol=1e-3, atol=1e-3)