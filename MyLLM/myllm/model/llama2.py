import torch 
import torch.nn as nn 
import torch.nn.functional as F  

import sentencepiece 

LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "num_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
}

class llama2(nn.Module):
    def __init__(self, config):
        pass 

    def forward(self , x):
        pass 


class Transformer(nn.Module):
    def __init__(self,config):
        pass


    def forward(self, x):
        pass


class RMSNorm(nn.Module):
    def __init__(self, config): 
        pass 

    def forward(self, x):
        pass 


class FlashAttention(nn.Module):
    def __init__(self, config):
        pass 


    def forward(self , x):
        pass


class FeedForward(nn.Module):
    def __init__(self, config):
        pass
    
    def forward(self, x):
        pass 




    

