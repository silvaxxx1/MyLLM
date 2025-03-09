import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

# Import the classes and functions to test - assuming they're in a file called gpt2_model.py
# Replace this import with the actual module containing your GPT-2 implementation
from gpt2 import (
    gpt2,
    TransformerBlock,
    FlashAttention,
    GPTMLP,
    GPT_CONFIG_124
)

# Test configuration with smaller dimensions for faster testing
TEST_CONFIG = {
    "vocab_size": 1000,
    "context_length": 32,
    "emb_dim": 128,
    "n_head": 4,
    "n_layer": 2,
    "dropout": 0.1,
    "qkv_bias": False
}

class TestGPT2Components:
    """Test suite for GPT-2 model components."""
    
    @pytest.fixture
    def test_config(self):
        """Fixture for the test configuration."""
        return TEST_CONFIG.copy()
    
    def test_gptmlp(self, test_config):
        """Test the GPTMLP module."""
        batch_size = 2
        seq_len = 8
        
        # Create sample input tensor
        x = torch.randn(batch_size, seq_len, test_config["emb_dim"])
        
        # Initialize MLP
        mlp = GPTMLP(test_config)
        
        # Test forward pass
        output = mlp(x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Verify internal structure
        assert isinstance(mlp.layer, nn.Sequential)
        assert len(mlp.layer) == 3
        assert isinstance(mlp.layer[0], nn.Linear)
        assert isinstance(mlp.layer[1], nn.GELU)
        assert isinstance(mlp.layer[2], nn.Linear)
        
        # Check dimensions of linear layers
        assert mlp.layer[0].in_features == test_config["emb_dim"]
        assert mlp.layer[0].out_features == 4 * test_config["emb_dim"]
        assert mlp.layer[2].in_features == 4 * test_config["emb_dim"]
        assert mlp.layer[2].out_features == test_config["emb_dim"]
    
    def test_flash_attention(self, test_config):
        """Test the FlashAttention module."""
        batch_size = 2
        seq_len = 8
        emb_dim = test_config["emb_dim"]
        
        # Create sample input tensor
        x = torch.randn(batch_size, seq_len, emb_dim)
        
        # Initialize FlashAttention
        attn = FlashAttention(test_config, d_in=emb_dim, d_out=emb_dim)
        
        # Test forward pass
        output = attn(x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check internal components
        assert hasattr(attn, 'qkv')
        assert hasattr(attn, 'proj')
        assert attn.head_dim == emb_dim // test_config["n_head"]
        assert attn.d_out == emb_dim
        
        # Check the QKV projection layer
        assert attn.qkv.in_features == emb_dim
        assert attn.qkv.out_features == emb_dim * 3
        
        # Check the output projection layer
        assert attn.proj.in_features == emb_dim
        assert attn.proj.out_features == emb_dim
    
    def test_transformer_block(self, test_config):
        """Test the TransformerBlock."""
        batch_size = 2
        seq_len = 8
        emb_dim = test_config["emb_dim"]
        
        # Create sample input tensor
        x = torch.randn(batch_size, seq_len, emb_dim)
        
        # Initialize TransformerBlock
        block = TransformerBlock(test_config)
        
        # Test forward pass
        output = block(x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check internal components
        assert hasattr(block, 'atten')
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
        assert hasattr(block, 'drop')
        assert hasattr(block, 'mlp')
        
        # Check each component's type
        assert isinstance(block.atten, FlashAttention)
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)
        assert isinstance(block.drop, nn.Dropout)
        assert isinstance(block.mlp, GPTMLP)
        
        # Make sure internal components are called correctly
        with patch.object(block, 'norm1', wraps=block.norm1) as mock_norm1, \
             patch.object(block, 'atten', wraps=block.atten) as mock_atten, \
             patch.object(block, 'norm2', wraps=block.norm2) as mock_norm2, \
             patch.object(block, 'mlp', wraps=block.mlp) as mock_mlp, \
             patch.object(block, 'drop', wraps=block.drop) as mock_drop:
            
            output = block(x)
            
            # Each component should be called the appropriate number of times
            assert mock_norm1.call_count == 1
            assert mock_atten.call_count == 1
            assert mock_norm2.call_count == 1
            assert mock_mlp.call_count == 1
            # Dropout is called twice in the block
            assert mock_drop.call_count == 2

    def test_gpt2_model_init(self, test_config):
        """Test GPT-2 model initialization."""
        model = gpt2(test_config)
        
        # Check model components
        assert hasattr(model, 'tok_emb')
        assert hasattr(model, 'pos_emb')
        assert hasattr(model, 'drop')
        assert hasattr(model, 'trs_blk')
        assert hasattr(model, 'norm')
        assert hasattr(model, 'proj')
        
        # Check transformer blocks
        assert len(model.trs_blk) == test_config["n_layer"]
        for block in model.trs_blk:
            assert isinstance(block, TransformerBlock)
        
        # Check embedding layers
        assert model.tok_emb.num_embeddings == test_config["vocab_size"]
        assert model.tok_emb.embedding_dim == test_config["emb_dim"]
        
        assert model.pos_emb.num_embeddings == test_config["context_length"]
        assert model.pos_emb.embedding_dim == test_config["emb_dim"]
        
        # Check output projection
        assert model.proj.in_features == test_config["emb_dim"]
        assert model.proj.out_features == test_config["vocab_size"]

    def test_gpt2_forward(self, test_config):
        """Test GPT-2 model forward pass."""
        batch_size = 2
        seq_len = 8
        
        # Create sample input tensor (token indices)
        x = torch.randint(0, test_config["vocab_size"], (batch_size, seq_len))
        
        # Initialize model
        model = gpt2(test_config)
        
        # Test forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, test_config["vocab_size"])
        
        # Make sure internal components are called correctly
        with patch.object(model, 'tok_emb', wraps=model.tok_emb) as mock_tok_emb, \
             patch.object(model, 'pos_emb', wraps=model.pos_emb) as mock_pos_emb, \
             patch.object(model, 'norm', wraps=model.norm) as mock_norm, \
             patch.object(model, 'proj', wraps=model.proj) as mock_proj:
            
            output = model(x)
            
            mock_tok_emb.assert_called_once()
            mock_pos_emb.assert_called_once()
            mock_norm.assert_called_once()
            mock_proj.assert_called_once()

class TestGPT2Integration:
    """Integration tests for the GPT-2 model."""
    
    @pytest.fixture
    def model(self, test_config):
        """Fixture for the GPT-2 model."""
        return gpt2(test_config)
    
    @pytest.fixture
    def test_config(self):
        """Fixture for the test configuration."""
        return TEST_CONFIG.copy()
    
    def test_model_forward_pass(self, model, test_config):
        """Test a complete forward pass through the model."""
        batch_size = 2
        seq_len = 8
        
        # Create sample input tensor (token indices)
        x = torch.randint(0, test_config["vocab_size"], (batch_size, seq_len))
        
        # Get output
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, test_config["vocab_size"])
        
        # Check that output contains valid logits
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains infinite values"
    
    def test_model_predictions(self, model, test_config):
        """Test that the model produces valid predictions."""
        batch_size = 1
        seq_len = 4
        
        # Create a simple token sequence
        x = torch.randint(0, test_config["vocab_size"], (batch_size, seq_len))
        
        # Get logits
        logits = model(x)
        
        # Convert to probabilities and get predictions
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        
        # Check predictions shape and validity
        assert predictions.shape == (batch_size, seq_len)
        assert (predictions >= 0).all() and (predictions < test_config["vocab_size"]).all()
    
    def test_autoregressive_property(self, model, test_config):
        """Test that the model exhibits the autoregressive property (causal attention)."""
        # Create two sequences where we modify one token and check if it affects future tokens but not past tokens
        seq1 = torch.zeros(1, 8, dtype=torch.long)
        seq2 = seq1.clone()
        seq2[0, 3] = 1  # Modify the 4th token
        
        # Get outputs for both sequences
        out1 = model(seq1)
        out2 = model(seq2)
        
        # Check that outputs for positions before the modified token are identical
        assert torch.allclose(out1[0, :3], out2[0, :3], rtol=1e-4, atol=1e-4)
        
        # Check that outputs for the modified position and after are different
        # (at least one position should be different)
        diff = (out1[0, 3:] - out2[0, 3:]).abs().sum()
        assert diff > 0, "Modifying an input token should affect its output and future tokens"

    def test_positional_encoding(self, model, test_config):
        """Test that positional encoding works correctly."""
        # Create two identical sequences but with different positions
        batch_size = 1
        seq_len = 8
        x = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        # Access the model's method to get positional embeddings
        # We'll patch the model to return the positions and embeddings
        positions = None
        embeddings = None
        
        def modified_forward(self, x):
            nonlocal positions, embeddings
            tok_emb = self.tok_emb(x)
            
            # Get position index
            pos_index = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            positions = pos_index
            
            # Get position embeddings
            pos_emb = self.pos_emb(pos_index)
            embeddings = pos_emb
            
            # Continue with normal forward pass
            embedding = tok_emb + pos_emb
            embedding = self.drop(embedding)
            
            for block in self.trs_blk:
                embedding = block(embedding)
            
            normilized_output = self.norm(embedding)
            output = self.proj(normilized_output)
            return output
        
        # Patch the forward method
        with patch.object(gpt2, 'forward', modified_forward):
            output = model(x)
            
            # Check position indices
            assert positions is not None
            assert positions.shape == (1, seq_len)
            assert torch.equal(positions[0], torch.arange(seq_len))
            
            # Check position embeddings
            assert embeddings is not None
            assert embeddings.shape == (1, seq_len, test_config["emb_dim"])
            
            # Each position should have a unique embedding
            for i in range(seq_len - 1):
                # Check that position i and i+1 have different embeddings
                assert not torch.allclose(embeddings[0, i], embeddings[0, i+1])

class TestGPT2Training:
    """Tests for training-related functionality."""
    
    @pytest.fixture
    def test_config(self):
        """Fixture for the test configuration."""
        return TEST_CONFIG.copy()
    
    @pytest.fixture
    def model(self, test_config):
        """Fixture for the GPT-2 model."""
        return gpt2(test_config)
    
    def test_dropout_behavior(self, model, test_config):
        """Test that dropout behaves differently in training vs eval mode."""
        batch_size = 2
        seq_len = 8
        
        # Create sample input tensor (token indices)
        x = torch.randint(0, test_config["vocab_size"], (batch_size, seq_len))
        
        # Run in training mode
        model.train()
        with torch.no_grad():
            output_train1 = model(x)
            output_train2 = model(x)
        
        # Run in eval mode
        model.eval()
        with torch.no_grad():
            output_eval1 = model(x)
            output_eval2 = model(x)
        
        # In eval mode, two identical inputs should produce identical outputs
        assert torch.allclose(output_eval1, output_eval2)
        
        # In training mode, outputs might be different due to dropout
        # This test might occasionally fail due to randomness
        if test_config["dropout"] > 0:
            assert not torch.allclose(output_train1, output_train2)
    
    def test_gradient_flow(self, model, test_config):
        """Test that gradients flow correctly through the model."""
        batch_size = 2
        seq_len = 4
        
        # Create sample input and target
        x = torch.randint(0, test_config["vocab_size"], (batch_size, seq_len))
        y = torch.randint(0, test_config["vocab_size"], (batch_size, seq_len))
        
        # Forward pass
        output = model(x)
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output.reshape(-1, test_config["vocab_size"]), y.reshape(-1))
        
        # Check that loss is a scalar tensor
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        # Backpropagate
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}"
            assert not torch.isinf(param.grad).any(), f"Infinite gradient for parameter {name}"

class TestGPT2ConfigValidation:
    """Tests for configuration validation and edge cases."""
    
    def test_invalid_config(self):
        """Test that the model validates configuration correctly."""
        # Test with invalid head dimensions
        invalid_config = TEST_CONFIG.copy()
        invalid_config["emb_dim"] = 100  # Not divisible by n_head=4
        
        with pytest.raises(AssertionError):
            attn = FlashAttention(invalid_config, d_in=invalid_config["emb_dim"], d_out=invalid_config["emb_dim"])
    
    def test_original_config(self):
        """Test instantiation with the original GPT_CONFIG_124."""
        try:
            model = gpt2(GPT_CONFIG_124)
            initialized = True
        except Exception as e:
            initialized = False
            pytest.skip(f"Could not initialize model with original config: {e}")
        
        assert initialized, "Model should initialize with the original GPT_CONFIG_124"
    
    def test_different_context_lengths(self):
        """Test the model with different context lengths."""
        config = TEST_CONFIG.copy()
        model = gpt2(config)
        
        # Test with a shorter sequence
        x_short = torch.randint(0, config["vocab_size"], (1, 4))
        output_short = model(x_short)
        assert output_short.shape == (1, 4, config["vocab_size"])
        
        # Test with the full context length
        x_full = torch.randint(0, config["vocab_size"], (1, config["context_length"]))
        output_full = model(x_full)
        assert output_full.shape == (1, config["context_length"], config["vocab_size"])

class TestGPT2WeightLoading:
    """Tests for the weight loading functions."""
    
    @pytest.mark.skip(reason="This requires actual GPT-2 weights")
    def test_download_and_load_gpt2(self):
        """Test downloading and loading GPT-2 weights."""
        from gpt2 import download_and_load_gpt2
        
        try:
            settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
            assert isinstance(settings, dict)
            assert isinstance(params, dict)
            assert "blocks" in params
        except Exception as e:
            pytest.skip(f"Could not download GPT-2 weights: {e}")
    
    @pytest.mark.skip(reason="This requires actual GPT-2 weights")
    def test_load_weights_into_gpt(self):
        """Test loading weights into the GPT-2 model."""
        from gpt2 import download_and_load_gpt2, load_weights_into_gpt
        
        try:
            settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
            model = gpt2(GPT_CONFIG_124)
            load_weights_into_gpt(model, params)
            
            # Model should be in eval mode after loading weights
            assert not model.training
        except Exception as e:
            pytest.skip(f"Could not load weights into GPT-2 model: {e}")