import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import torch
from GPT import GPTModel, TransformerBlock, MultiHeadAttention, LayerNorm, FeedForward



class TestGPTModel(unittest.TestCase):
    def setUp(self):
        # Define a minimal configuration for testing
        self.cfg = {
            "vocab_size": 100,         # Small vocab size for testing
            "emb_dim": 32,             # Embedding dimension
            "context_length": 10,      # Short context length
            "n_heads": 4,              # Number of attention heads
            "drop_rate": 0.1,          # Dropout rate
            "n_layers": 2,             # Number of transformer layers
            "qkv_bias": False          # No bias for qkv projections
        }
        # Initialize the model
        self.model = GPTModel(self.cfg)

    def test_forward_shape(self):
        """Test that the model's forward pass outputs the correct shape."""
        batch_size = 8
        seq_len = self.cfg["context_length"]
        
        # Create a batch of random token indices (in the range of vocab_size)
        input_data = torch.randint(0, self.cfg["vocab_size"], (batch_size, seq_len))

        # Forward pass through the model
        output = self.model(input_data)

        # Check the output shape: (batch_size, seq_len, vocab_size)
        self.assertEqual(output.shape, (batch_size, seq_len, self.cfg["vocab_size"]))

    def test_embedding_layer(self):
        """Test that the token and positional embeddings have the correct shapes."""
        batch_size = 4
        seq_len = self.cfg["context_length"]

        # Create input token indices
        input_data = torch.randint(0, self.cfg["vocab_size"], (batch_size, seq_len))

        # Test the token embedding layer
        tok_emb = self.model.tok_emb(input_data)
        self.assertEqual(tok_emb.shape, (batch_size, seq_len, self.cfg["emb_dim"]))

        # Test the positional embedding layer
        pos_emb = self.model.pos_emb(torch.arange(seq_len, device=input_data.device))
        self.assertEqual(pos_emb.shape, (seq_len, self.cfg["emb_dim"]))

    def test_transformer_block_forward(self):
        """Test that the transformer block returns the correct shape."""
        transformer_block = TransformerBlock(self.cfg)
        batch_size = 4
        seq_len = self.cfg["context_length"]
        emb_dim = self.cfg["emb_dim"]

        # Create random input with shape (batch_size, seq_len, emb_dim)
        input_data = torch.rand(batch_size, seq_len, emb_dim)

        # Forward pass through a single transformer block
        output = transformer_block(input_data)

        # Check that the output shape matches the input shape
        self.assertEqual(output.shape, input_data.shape)

    def test_multihead_attention(self):
        """Test that the Multi-Head Attention mechanism produces the correct shape."""
        attention = MultiHeadAttention(
            d_in=self.cfg["emb_dim"], 
            d_out=self.cfg["emb_dim"], 
            context_length=self.cfg["context_length"], 
            dropout=0.1, 
            num_heads=self.cfg["n_heads"]
        )
        batch_size = 4
        seq_len = self.cfg["context_length"]
        emb_dim = self.cfg["emb_dim"]

        # Create random input with shape (batch_size, seq_len, emb_dim)
        input_data = torch.rand(batch_size, seq_len, emb_dim)

        # Forward pass through multi-head attention
        output = attention(input_data)

        # Check that the output shape matches the expected shape
        self.assertEqual(output.shape, (batch_size, seq_len, emb_dim))

    def test_layer_norm(self):
        """Test that LayerNorm produces the correct normalized output shape."""
        layer_norm = LayerNorm(self.cfg["emb_dim"])
        batch_size = 4
        seq_len = self.cfg["context_length"]
        emb_dim = self.cfg["emb_dim"]

        # Create random input with shape (batch_size, seq_len, emb_dim)
        input_data = torch.rand(batch_size, seq_len, emb_dim)

        # Apply layer normalization
        output = layer_norm(input_data)

        # Check that the output shape matches the input shape
        self.assertEqual(output.shape, input_data.shape)

    def test_feed_forward(self):
        """Test that the feed-forward network produces the correct output shape."""
        feed_forward = FeedForward(self.cfg)
        batch_size = 4
        seq_len = self.cfg["context_length"]
        emb_dim = self.cfg["emb_dim"]

        # Create random input with shape (batch_size, seq_len, emb_dim)
        input_data = torch.rand(batch_size, seq_len, emb_dim)

        # Forward pass through the feed-forward network
        output = feed_forward(input_data)

        # Check that the output shape matches the input shape
        self.assertEqual(output.shape, (batch_size, seq_len, emb_dim))


if __name__ == "__main__":
    unittest.main()
