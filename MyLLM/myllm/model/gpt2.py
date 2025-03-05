from config import Config 
import torch
import torch.nn as nn


GPT_CONFIG_124 = {
    "vocab_size": 50257,      # Size of the vocabulary
    "context_length": 1024,   # Maximum length of the input context
    "emb_dim": 768,           # Dimension of the embeddings
    "n_head": 12,             # Number of attention heads
    "n_layer": 12,            # Number of transformer layers
    "dropout": 0.1,           # Dropout rate for regularization
    "qkv_bias": False,        # Whether to include bias in Q, K, V projections
}

# device handling
device = "cuda" if torch.cuda.is_available else "cpu"

class gpt2(nn.Module):
    def __init__(self,config):
        super(gpt2, self).__init__()

        self.tok_emb = nn.Embedding(config["vocab_size"] , config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"] , config["emb_dim"]) 
        self.drop = nn.Dropout(config["dropout"]) 
        self.trs_blk = nn.Sequential(
            [*TransformerBlock[config] for _ in range(config['n_layer'])]
        )
        self.norm = nn.LayerNorm(config["emb_dim"])
        self.proj = nn.Linear(config['emb_dim'], config['vocab_size']) 

    def forward(self,x):
        # x (input) = (batch_size , sequence_len)

        # (batch_size , sequence_len) --> (batch_size , sequence_len,emb_dim)
        tok_emb = self.tok_emb(x)
        # take the input x = batch_size , sequence_len) 
        # x.size(1) (sequence_len) 
        # (sequence_len) unsqueeze --> (1 , )sequence_len
        pos_index = torch.arange(x.size(1) , device=device).unsqueeze(0)
        # (batch_size , sequence_len) --> (batch_size , sequence_len,emb_dim)
        pos_emb = self.pos_emb(pos_index)
        # (batch_size , sequence_len,emb_dim) + (batch_size , sequence_len,emb_dim) --> (batch_size , sequence_len,emb_dim)
        embedding = tok_emb + pos_emb
        # dropout (dims remain the same)
        embedding = self.drop(embedding)
        # (batch_size , sequence_len,emb_dim) --> (batch_size , sequence_len,emb_dim) 
        transformer_output = self.trs_blk(embedding)
        # norm the transformer output 
        normilized_output = self.norm(transformer_output)
        # (batch_size , sequence_len,emb_dim) --> (batch_size , sequence_len,emb_dim) 
        output = self.proj(normilized_output)

        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()


class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__() 


class GPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()


class 












        