import torch
import torch.nn.functional as F

def top_p_logits(logits, p=0.5):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumulative_probs > p] = 0
    filtered_logits = torch.zeros_like(logits).to(logits.device)  # Move to same device
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    return filtered_logits

def generate(model, prompt, max_new_tokens, context_size, tokenizer, temperature=0.0, top_k=None, top_p=None, eos=None):
    # Detect device
    device = next(model.parameters()).device  # Get model's device

    # Encode and move input to the correct device
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    idx_gen = idx.clone()  # Start with the prompt indices

    for _ in range(max_new_tokens):
        idx_cond = idx_gen[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)  # Forward pass
            logits = logits[:, -1, :]  # Take the last token's logits
            
            # Apply top-k sampling
            if top_k is not None:
                top_k_values, _ = torch.topk(logits, k=top_k)
                min_value = top_k_values[:, -1].unsqueeze(1)  
                logits = torch.where(logits < min_value, torch.tensor(float('-inf')).to(device), logits)

            # Apply top-p sampling
            if top_p is not None:
                logits = top_p_logits(logits, p=top_p)  

            # Apply temperature
            if temperature > 0.0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)  # Convert to probabilities
                idx_next = torch.multinomial(probs, num_samples=1)  # Sample token
            else: 
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Take max logit

            # EOS handling
            if eos is not None and torch.equal(idx_next, torch.tensor(eos).to(device)):
                break
            
            # Append new token
            idx_gen = torch.cat((idx_gen, idx_next), dim=1)

    return tokenizer.decode(idx_gen.squeeze(0).tolist())  # Convert tokens back to text
