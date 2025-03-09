import torch 
import torch.nn as nn 
import torch.nn.functional as F  

import sentencepiece 

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        self.token_emb = nn.Embedding(config["vocab_size"], config["emb_dim"],dtype=config["dtype"]) 
        self.trs_blk = nn.ModuleList(Transformer(config) for _ in range(config['n_layer'])) 
        self.norm = nn.LayerNorm(config["emb_dim"]) 
        self.proj = nn.Linear(config['emb_dim'], config['vocab_size'], dtype=config["dtype"] , bias=False)

    def forward(self , x):
        x = self.token_emb(x) 
        x = self.trs_blk(x) 
        x = self.norm(x)
        logits = self.proj(x)

        return logits



class Transformer(nn.Module):
    def __init__(self,config):
        self.norm1 = nn.RMSNorm(config["emb_dim"], eps=1e-5)
        self.norm2 = nn.RMSNorm(config["emb_dim"], eps=1e-5)
        self.atten = FlashAttention(config) 
        


    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.atten(x)
        x = self.norm2(x)
        x = shortcut + x

        return x



class FlashAttention(nn.Module):
    def __init__(self, config):
        pass


    def forward(self , x):
        pass



class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)




    

