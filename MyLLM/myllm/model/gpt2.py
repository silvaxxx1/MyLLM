import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration for the GPT-2 model (GPT_CONFIG_124) based on the original GPT-2 architecture.
GPT_CONFIG_124 = {
    "vocab_size": 50257,      # Size of the vocabulary (number of tokens)
    "context_length": 1024,   # Maximum length of the input context
    "emb_dim": 768,           # Dimension of the token and positional embeddings
    "n_head": 12,             # Number of attention heads in the multi-head attention mechanism
    "n_layer": 12,            # Number of transformer layers (blocks)
    "dropout": 0.1,           # Dropout rate for regularization
    "qkv_bias": False,        # Whether to include bias terms in the query, key, value projections
}

# Device handling for model training/inference on GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

class gpt2(nn.Module):
    """
    GPT-2 model implementation. A transformer-based language model for text generation.
    The model consists of token and positional embeddings, transformer blocks, layer normalization,
    and an output projection to predict the next token.
    """
    def __init__(self, config):
        """
        Initialize the GPT-2 model.
        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super(gpt2, self).__init__()

        # Token and position embeddings
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"]) 
        
        # Dropout layer for regularization
        self.drop = nn.Dropout(config["dropout"]) 
        
        # List of transformer blocks (stacked layers)
        self.trs_blk = nn.ModuleList([TransformerBlock(config) for _ in range(config['n_layer'])])
        
        # Layer normalization and final linear projection to vocab size
        self.norm = nn.LayerNorm(config["emb_dim"])
        self.proj = nn.Linear(config['emb_dim'], config['vocab_size']) 

    def forward(self, x):
        """
        Forward pass of the GPT-2 model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_len) representing token IDs.
        Returns:
            torch.Tensor: Output logits of shape (batch_size, sequence_len, vocab_size).
        """
        # Token embedding (batch_size, sequence_len) --> (batch_size, sequence_len, emb_dim)
        tok_emb = self.tok_emb(x)
        
        # Positional embedding (batch_size, sequence_len) --> (batch_size, sequence_len, emb_dim)
        pos_index = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos_index)
        
        # Add token and position embeddings together
        embedding = tok_emb + pos_emb
        
        # Apply dropout
        embedding = self.drop(embedding)
        
        # Pass through transformer blocks
        for block in self.trs_blk:
            embedding = block(embedding)  # Apply each transformer block
        
        # Normalize the output of the transformer
        normilized_output = self.norm(embedding)
        
        # Project the output back to the vocabulary size
        output = self.proj(normilized_output)

        return output

    
class TransformerBlock(nn.Module):
    """
    A single transformer block containing attention and feed-forward layers with residual connections.
    """
    def __init__(self, config):
        """
        Initialize the transformer block.
        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super().__init__()

        # Attention mechanism, layer normalization, dropout, and feed-forward network
        self.atten = FlashAttention(config,d_in=config["emb_dim"], d_out=config["emb_dim"])
        self.norm1 = nn.LayerNorm(config["emb_dim"])
        self.norm2 = nn.LayerNorm(config["emb_dim"])
        self.drop = nn.Dropout(config["dropout"]) 
        self.mlp = GPTMLP(config)

    def forward(self, x):
        """
        Forward pass through the transformer block.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_len, emb_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_len, emb_dim).
        """
        shortcut = x 
        
        # Apply the first layer normalization and attention
        x = self.norm1(x)
        x = self.atten(x)
        
        # Apply dropout and residual connection
        x = self.drop(x)
        x = shortcut + x

        shortcut = x
        
        # Apply the second layer normalization and MLP (feed-forward network)
        x = self.norm2(x)
        x = self.mlp(x)
        
        # Apply dropout and residual connection
        x = self.drop(x)
        x = shortcut + x

        return x

class FlashAttention(nn.Module):
    """
    FlashAttention mechanism for efficient multi-head self-attention.
    """
    def __init__(self, config, d_in, d_out, qkv_bias=False, drop=0.0):
        """
        Initialize the FlashAttention mechanism.
        Args:
            config (dict): Configuration dictionary containing model parameters.
            d_in (int): Input dimensionality.
            d_out (int): Output dimensionality (should be divisible by the number of heads).
            qkv_bias (bool): Whether to include biases in the Q, K, V projections.
            drop (float): Dropout rate for attention scores.
        """
        super().__init__()

        assert d_out % config["n_head"] == 0, "embed_dim is indivisible by num_heads"
        self.head_dim = d_out // config["n_head"] 
        self.d_out = d_out 
        self.qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.drop = drop

    def forward(self, x):
        """
        Forward pass through FlashAttention.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, emb_dim).
        """
        batch_size, seq_len, emb_dims = x.shape
        
        # Project input into Q, K, and V
        qkv = self.qkv(x)
        
        # Reshape Q, K, V to (batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, seq_len, 3, config["n_head"], self.head_dim)
        
        # Rearrange dimensions for multi-head attention
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        queries, keys, values = qkv 

        # Use dropout if the model is in training mode
        use_dropout = 0. if not self.training else self.drop

        # Perform FlashAttention (scaled dot-product attention)
        out = F.scaled_dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            attn_mask=None,
            dropout_p=use_dropout,
            is_causal=True
        )

        # Combine heads and reshape back to (batch_size, seq_len, d_out)
        context_vec = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)

        # Output projection
        context_vec = self.proj(context_vec)

        return context_vec

class GPTMLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) for use in GPT-2's feed-forward layers.
    """
    def __init__(self, config):
        """
        Initialize the GPT MLP.
        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super().__init__()

        # A sequential MLP consisting of a linear layer, GELU activation, and another linear layer
        self.layer = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),  # Projection to 4 times the embedding size
            nn.GELU(),  # GELU activation function
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),  # Back to embedding size
        )

    def forward(self, x):
        """
        Forward pass through the GPT MLP.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, emb_dim).
        """
        return self.layer(x)




# Use the provided GPT_CONFIG_124 or create a configuration dictionary
config = GPT_CONFIG_124

