import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

# Import the model (assuming the file is named gpt2_model.py)
# Replace with your actual import path
from gpt2 import gpt2, TransformerBlock, FlashAttention, GPTMLP, GPT_CONFIG_124

@pytest.fixture
def config():
    return GPT_CONFIG_124

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class TestGPT2:
    def test_model_init(self, config):
        """Test if the model can be initialized correctly."""
        model = gpt2(config)
        # Check if the model has the expected components
        assert isinstance(model.tok_emb, nn.Embedding)
        assert isinstance(model.pos_emb, nn.Embedding)
        assert isinstance(model.drop, nn.Dropout)
        assert len(model.trs_blk) == config["n_layer"]
        assert isinstance(model.norm, nn.LayerNorm)
        assert isinstance(model.proj, nn.Linear)
        
    def test_model_forward(self, config, device):
        """Test the forward pass of the model."""
        model = gpt2(config).to(device)
        batch_size = 2
        seq_len = 10
        
        # Create a sample input
        x = torch.randint(0, config["vocab_size"], (batch_size, seq_len)).to(device)
        
        # Run forward pass
        out = model(x)
        
        # Check output shape
        assert out.shape == (batch_size, seq_len, config["vocab_size"])
        
    def test_transformer_block(self, config, device):
        """Test the transformer block."""
        block = TransformerBlock(config).to(device)
        batch_size = 2
        seq_len = 10
        
        # Create a sample input
        x = torch.randn(batch_size, seq_len, config["emb_dim"]).to(device)
        
        # Run forward pass
        out = block(x)
        
        # Check output shape
        assert out.shape == (batch_size, seq_len, config["emb_dim"])
        
    def test_flash_attention(self, config, device):
        """Test the FlashAttention mechanism."""
        attention = FlashAttention(config, config["emb_dim"], config["emb_dim"]).to(device)
        batch_size = 2
        seq_len = 10
        
        # Create a sample input
        x = torch.randn(batch_size, seq_len, config["emb_dim"]).to(device)
        
        # Run forward pass
        out = attention(x)
        
        # Check output shape
        assert out.shape == (batch_size, seq_len, config["emb_dim"])
        
    def test_gpt_mlp(self, config, device):
        """Test the GPT MLP."""
        mlp = GPTMLP(config).to(device)
        batch_size = 2
        seq_len = 10
        
        # Create a sample input
        x = torch.randn(batch_size, seq_len, config["emb_dim"]).to(device)
        
        # Run forward pass
        out = mlp(x)
        
        # Check output shape
        assert out.shape == (batch_size, seq_len, config["emb_dim"])
        
    def test_embedding_dimensions(self, config, device):
        """Test that embeddings have the correct dimensions."""
        model = gpt2(config).to(device)
        
        assert model.tok_emb.weight.shape == (config["vocab_size"], config["emb_dim"])
        assert model.pos_emb.weight.shape == (config["context_length"], config["emb_dim"])
        
    def test_autoregressive_property(self, config, device):
        """Test that the model respects autoregressive property in attention."""
        model = gpt2(config).to(device)
        batch_size = 2
        seq_len = 10
        
        # Create identical inputs
        x1 = torch.randint(0, config["vocab_size"], (batch_size, seq_len)).to(device)
        x2 = x1.clone()
        
        # Change one token in the middle
        change_pos = seq_len // 2
        if x2[0, change_pos] < config["vocab_size"] - 1:
            x2[0, change_pos] += 1
        else:
            x2[0, change_pos] -= 1
            
        # Get outputs for both inputs
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
            
        # Check that positions before change_pos are identical, positions after may differ
        assert_close(out1[:, :change_pos, :], out2[:, :change_pos, :])
        
        # Check that at least the changed position output is different
        assert not torch.allclose(out1[:, change_pos:, :], out2[:, change_pos:, :], rtol=1e-3, atol=1e-3)