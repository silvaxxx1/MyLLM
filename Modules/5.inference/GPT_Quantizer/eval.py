import torch 
import math
from torch.utils.data import DataLoader


# Perplexity calculation function
def calculate_perplexity(model, tokenizer, input_text, device='cpu'):
    
    # 1. Tokenize the input text
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)  # Add batch dimension

    # 2. Put model in evaluation mode
    model.eval()

    # 3. Get the model's output (logits) for the input text
    with torch.no_grad():
        logits = model(input_ids)  # Shape: (batch_size, seq_length, vocab_size)
        
    # 4. Calculate the log-likelihood of the target tokens
    shift_logits = logits[:, :-1, :].contiguous()  # Shift logits by 1 for next token prediction
    shift_labels = input_ids[:, 1:].contiguous()  # Shift labels to match the target
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')  # No reduction to get loss per token
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # 5. Calculate the perplexity (exponential of the average loss)
    avg_loss = loss.mean().item()  # Average loss
    perplexity = math.exp(avg_loss)  # Perplexity is exp of average loss

    return perplexity
