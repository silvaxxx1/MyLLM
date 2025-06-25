# 🧠 Step-by-Step GPT-Style Transformer Block Walkthrough

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import tiktoken 

# 🔌 Setup (force CPU if CUDA misconfigured)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

# ========================================
# 1. 🔤 Tokenization
# ========================================
text = "My name is Silva"
tok = tiktoken.get_encoding("gpt2")
token_ids = tok.encode(text)
input_tensor = torch.tensor(token_ids).unsqueeze(0)  # shape: (1, seq_len)
print(f"Input tensor shape: {input_tensor.shape}")

# ========================================
# 2. 🔠 Token + Positional Embeddings
# ========================================
emd_dims = 256
vocab_size = 50257
max_seq_len = 512

emd_layer = nn.Embedding(vocab_size, emd_dims)
emb_output = emd_layer(input_tensor)
print(f"Token embedding shape: {emb_output.shape}")

pos_emb = nn.Embedding(max_seq_len, emd_dims)
seq_len = input_tensor.shape[1]
pos_ids = torch.arange(seq_len).unsqueeze(0)
pos_emb_output = pos_emb(pos_ids)
print(f"Positional embedding shape: {pos_emb_output.shape}")

combined_emb = emb_output + pos_emb_output
print(f"Combined embedding shape: {combined_emb.shape}")

# ========================================
# 3. 🧠 Multi-Head Self Attention
# ========================================
head_size = 64
num_head = 8
assert emd_dims % num_head == 0

# Project to Q, K, V
W_qkv = nn.Linear(emd_dims, 3 * head_size * num_head)
qkv = W_qkv(combined_emb)
q, k, v = qkv.chunk(3, dim=-1)
print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

# Reshape into heads
q = q.view(1, seq_len, num_head, head_size).transpose(1, 2)
k = k.view(1, seq_len, num_head, head_size).transpose(1, 2)
v = v.view(1, seq_len, num_head, head_size).transpose(1, 2)
print(f"After split & transpose: Q shape: {q.shape}")

# ========================================
# 4. 🚫 Causal Masking
# ========================================
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
mask = mask.unsqueeze(0).unsqueeze(0).expand(1, num_head, seq_len, seq_len)
print(f"Mask shape: {mask.shape}")

# ========================================
# 5. 📏 Attention Calculation
# ========================================
scores = torch.matmul(q, k.transpose(-1, -2)) / (head_size ** 0.5)
scores = scores.masked_fill(mask, float('-inf'))
scores = F.softmax(scores, dim=-1)
output = torch.matmul(scores, v)
print(f"Attention output shape (pre-merge): {output.shape}")

# ========================================
# 6. 🧩 Merge Heads + Project
# ========================================
output = output.transpose(1, 2).contiguous().view(1, seq_len, head_size * num_head)
W_out = nn.Linear(head_size * num_head, emd_dims)
output = W_out(output)
print(f"Attention output shape (post-project): {output.shape}")

# ========================================
# 7. 🔁 Residual + LayerNorm + MLP
# ========================================
x = combined_emb + output  # Residual connection
ln2 = nn.LayerNorm(emd_dims)
x_ln = ln2(x)

mlp = nn.Sequential(
    nn.Linear(emd_dims, 4 * emd_dims),
    nn.GELU(),
    nn.Linear(4 * emd_dims, emd_dims)
)
mlp_output = mlp(x_ln)
x = x + mlp_output  # Final residual
print(f"Final output shape: {x.shape}")  # (1, seq_len, emd_dims)
