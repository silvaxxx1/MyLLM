import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout_rate=0.1, bias=False):
        super().__init__()
        # Ensure output dimension is divisible by the number of heads
        assert (d_out % num_heads) == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Initialize linear layers for query, key, and value transformations
        self.W_query = nn.Linear(d_in, d_out, bias=bias)  # Linear layer for queries
        self.W_key = nn.Linear(d_in, d_out, bias=bias)    # Linear layer for keys
        self.W_value = nn.Linear(d_in, d_out, bias=bias)  # Linear layer for values

        self.dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(d_out, d_out)

        # Create an upper triangular mask to prevent information leakage
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # Apply linear transformations to the input x to obtain keys, values, and queries
        b, num_tokens, d_in = x.shape  # Input shape: (batch_size, num_tokens, d_in)

        keys = self.W_key(x)  # Shape: (batch_size, num_tokens, d_out)
        values = self.W_value(x)  # Shape: (batch_size, num_tokens, d_out)
        query = self.W_query(x)  # Shape: (batch_size, num_tokens, d_out)

        # Reshape and transpose for multi-head attention
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_tokens, head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_tokens, head_dim)
        query = query.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_tokens, head_dim)

        # Compute attention scores
        attention_score = query @ keys.transpose(2, 3)  # Shape: (batch_size, num_heads, num_tokens, num_tokens)

        # Apply mask to attention scores
        mask_bool = self.mask[:num_tokens, :num_tokens].bool()  # Shape: (num_tokens, num_tokens)
        attention_score.masked_fill_(mask_bool, -torch.inf)  # Shape remains: (batch_size, num_heads, num_tokens, num_tokens)

        # Calculate attention weights
        attention_weight = torch.softmax(attention_score / keys.shape[-1] ** 0.5, dim=-1)  # Shape: (batch_size, num_heads, num_tokens, num_tokens)
        attention_weight = self.dropout(attention_weight)

        # Calculate context vector
        all_con_vec = (attention_weight @ values)  # Shape: (batch_size, num_heads, num_tokens, head_dim)
        all_con_vec = all_con_vec.transpose(1, 2)  # Shape: (batch_size, num_tokens, num_heads, head_dim)
        all_con_vec = all_con_vec.contiguous().view(b, num_tokens, self.d_out)  # Shape: (batch_size, num_tokens, d_out)

        # Project the output
        output = self.proj(all_con_vec)  # Shape: (batch_size, num_tokens, d_out)
        return output


class MHACombinedQKV(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_head, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)

        # (b, num_heads, num_tokens, head_dim) --> (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**-0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, num_tokens) --> (b, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values

        # (b, num_heads, num_tokens, head_dim) --> (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)

        # (b, num_tokens, num_heads, head_dim) --> (b, num_tokens, embed_dim)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, embed_dim)

        context_vec = self.proj(context_vec)

        return context_vec

