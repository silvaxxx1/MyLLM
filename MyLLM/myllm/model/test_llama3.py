import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

# Import the model (assuming the file is named llama3_model.py)
# Replace with your actual import path
from llama3 import Llama3, TransformerBlock, GroupedQueryFlashAttention, FeedForward, LLAMA32_CONFIG

@pytest.fixture
def config():
    return LLAMA32_CONFIG

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class TestLlama3:
    def test_model_init(self, config):
        """Test if the model can be initialized correctly."""
        # Use smaller config for testing
        test_config = config.copy()
        test_config["n_layers"] = 2
        test_config["emb_dim"] = 1024
        test_config["hidden_dim"] = 2048
        test_config["n_heads"] = 8
        test_config["n_kv_groups"] = 4
        
        model = Llama3(test_config)
        
        # Check if the model has the expected components
        assert isinstance(model.tok_emb, nn.Embedding)
        assert len(model.trf_blocks) == test_config["n_layers"]
        assert isinstance(model.final_norm, nn.RMSNorm)
        assert isinstance(model.out_head, nn.Linear)
        
    def test_model_forward(self, config, device):
        """Test the forward pass of the model."""
        # Use smaller config for testing
        test_config = config.copy()
        test_config["n_layers"] = 2
        test_config["emb_dim"] = 1024
        test_config["hidden_dim"] = 2048
        test_config["n_heads"] = 8
        test_config["n_kv_groups"] = 4
        
        model = Llama3(test_config).to(device)
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
        # Use smaller config for testing
        test_config = config.copy()
        test_config["emb_dim"] = 1024
        test_config["hidden_dim"] = 2048
        test_config["n_heads"] = 8
        test_config["n_kv_groups"] = 4
        
        block = TransformerBlock(test_config).to(device)
        batch_size = 2
        seq_len = 10
        
        # Create a sample input
        x = torch.randn(batch_size, seq_len, test_config["emb_dim"]).to(device)
        
        # Run forward pass
        out = block(x)
        
        # Check output shape
        assert out.shape == (batch_size, seq_len, test_config["emb_dim"])
        
    def test_grouped_query_flash_attention(self, config, device):
        """Test the Grouped Query Flash Attention mechanism."""
        # Use smaller config for testing
        test_config = config.copy()
        test_config["emb_dim"] = 1024
        test_config["n_heads"] = 8
        test_config["n_kv_groups"] = 4
        
        attention = GroupedQueryFlashAttention(
            test_config, 
            test_config["emb_dim"], 
            test_config["emb_dim"]
        ).to(device)
        
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
        # Use smaller config for testing
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
        
    def test_pre_compute_freq(self, config):
        """Test the RoPE frequency computation for Llama3."""
        from llama3 import pre_compute_freq
        
        # Use smaller context for testing
        test_config = config.copy()
        test_config["context_length"] = 128
        test_config["emb_dim"] = 1024
        test_config["n_heads"] = 8
        
        freqs = pre_compute_freq(test_config)
        
        # Check shape (should be context_length x (head_dim/2))
        head_dim = test_config["emb_dim"] // test_config["n_heads"]
        assert freqs.shape == (test_config["context_length"], head_dim // 2)
        
        # Check that frequencies are complex numbers
        assert freqs.dtype == torch.complex64 or freqs.dtype == torch.complex128
        
    def test_apply_rope(self, device):
        """Test the RoPE application function."""
        from llama3 import apply_rope
        
        batch_size = 2
        n_heads = 8
        seq_len = 10
        head_dim = 128
        
        # Create a sample input and frequencies
        x = torch.randn(batch_size, n_heads, seq_len, head_dim).to(device)
        freqs = torch.polar(torch.ones(seq_len, head_dim // 2), 
                           torch.randn(seq_len, head_dim // 2)).to(device)
        
        # Apply RoPE
        out = apply_rope(x, freqs)
        
        # Check output shape
        assert out.shape == (batch_size, n_heads, seq_len, head_dim)
        
    def test_grouped_query_shape(self, config, device):
        """Test the grouped query attention shapes and expansion."""
        # Use smaller config for testing
        test_config = config.copy()
        test_config["emb_dim"] = 1024
        test_config["n_heads"] = 8
        test_config["n_kv_groups"] = 4
        
        attention = GroupedQueryFlashAttention(
            test_config, 
            test_config["emb_dim"], 
            test_config["emb_dim"]
        ).to(device)
        
        batch_size = 2
        seq_len = 10
        
        # Check that n_heads is divisible by n_kv_groups
        assert test_config["n_heads"] % test_config["n_kv_groups"] == 0
        
        # Create a sample input
        x = torch.randn(batch_size, seq_len, test_config["emb_dim"]).to(device)
        
        # Access internal projections to check shapes
        q = attention.q_proj(x)
        k = attention.k_proj(x)
        v = attention.v_proj(x)
        
        head_dim = test_config["emb_dim"] // test_config["n_heads"]
        
        # Check projection shapes
        assert q.shape == (batch_size, seq_len, test_config["emb_dim"])
        assert k.shape == (batch_size, seq_len, test_config["n_kv_groups"] * head_dim)
        assert v.shape == (batch_size, seq_len, test_config["n_kv_groups"] * head_dim)
        
    def test_autoregressive_property(self, config, device):
        """Test that the model respects autoregressive property (causal masking)."""
        # Use smaller config for testing
        test_config = config.copy()
        test_config["n_layers"] = 2
        test_config["emb_dim"] = 1024
        test_config["hidden_dim"] = 2048
        test_config["n_heads"] = 8
        test_config["n_kv_groups"] = 4
        
        model = Llama3(test_config).to(device)
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
        assert not torch.allclose(out1[:, change_pos:, :], out2[:, change_pos:, :], rtol=1e-3, atol=1e-3)