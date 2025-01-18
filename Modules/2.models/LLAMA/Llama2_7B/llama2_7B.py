import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
from dataclasses import dataclass 
from typing import Optional
from llama2_7B_Config import LLAMA2_CONFIG_7B


LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "num_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
}

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps 
        self.scale  =nn.Parameter(torch.ones(emb_dim))
        self.emd_dim = emb_dim


    def forward(self, x):
        mean = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(mean + self.eps)
        output = (self.scale * x_norm).to(dtype=x.dtype) 
        return output
        

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        self.silu = SiLU()

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x = self.silu(x1) * x2        
        return self.fc3(x)
    
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


# Define the MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, seq_len, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.d_in = d_in  # Input dimension
        self.d_out = d_out  # Output dimension
        self.seq_len = seq_len  # Sequence length
        self.num_heads = num_heads  # Number of attention heads
        self.head_dim = d_out // num_heads  # Dimension of each head
        
        assert self.head_dim * num_heads == d_out, "Output dimension must be divisible by number of heads"
        
        # Linear layers for query, key, and value
        self.query = nn.Linear(d_in, d_out)
        self.key = nn.Linear(d_in, d_out)
        self.value = nn.Linear(d_in, d_out)
        
        # Output projection
        self.fc_out = nn.Linear(d_out, d_out)
        
        # Softmax for attention scores
        self.softmax = nn.Softmax(dim=-1)
        
        # Precompute rotary embeddings (not implemented here, adjust as needed)
        # self.freqs_complex = precompute_theta_pos_frequencies(...)
        
    def forward(self, x):
        B, T, D_in = x.shape  # Batch size, sequence length, input dimension
        
        assert T == self.seq_len, "Input sequence length must match the configured sequence length"
        assert D_in == self.d_in, "Input dimension must match the configured input dimension"
        
        # Linear projections of query, key, and value
        Q = self.query(x)  # (B, T, d_out)
        K = self.key(x)  # (B, T, d_out)
        V = self.value(x)  # (B, T, d_out)
        
        # Reshape for multi-head attention: (B, T, num_heads, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        
        # Scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-2, -1))  # (B, num_heads, T, T)
        attention = self.softmax(energy / math.sqrt(self.head_dim))  # Normalize energy
        
        # Apply attention weights to value
        out = torch.matmul(attention, V)  # (B, num_heads, T, head_dim)
        
        # Combine heads and pass through final linear layer
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_out)  # (B, T, d_out)
        out = self.fc_out(out)  # (B, T, d_out)
        
        return out
    

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(d_in = cfg["emb_dim"],
                                       d_out=cfg["emb_dim"],
                                        seq_len=cfg["context_length"],
                                        num_heads=cfg["num_heads"]
        )
        
        self.ff = FeedForward(cfg)
        ##########################################
        # self.norm1 = LayerNorm(cfg["emb_dim"])
        # self.norm2 = LayerNorm(cfg["emb_dim"])
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
        ########################################

        # self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x 
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        # x = self.drop_resid(x)
        x = self.norm2(x)
        x = self.ff(x)
        # x = self.drop_resid(x)
        x = x + shortcut
        return x

# class GPTModel(nn.Module):
class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        # self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[Transformer(cfg) for _ in range(cfg["n_layers"])])

        ################################### NEW ###################################
        # self.final_norm = LayerNorm(cfg["emb_dim"])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        ###########################################################################
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        # batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds  # + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits



