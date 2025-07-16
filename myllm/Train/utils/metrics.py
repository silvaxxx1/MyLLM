# metrics.pyimport torch
import torch 

def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float()
    return correct.sum() / correct.numel()

def perplexity(loss):
    return torch.exp(loss)

# Add more as needed
