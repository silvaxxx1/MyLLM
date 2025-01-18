import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def token_to_text(token_ids, tokenizer):
    """
    Convert a sequence of token IDs into human-readable text.

    Args:
        token_ids (torch.Tensor): The tensor containing the token IDs to decode.
        tokenizer (Tokenizer): The tokenizer used for decoding the token IDs.

    Returns:
        str: Decoded text from the token sequence.
    """
    # Remove the batch dimension by squeezing and convert the token IDs to a list for decoding
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())  


def generate_text(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Generate a sequence of tokens using the model.

    Args:
        model (nn.Module): The language model used for token generation.
        idx (torch.Tensor): Input tensor containing the initial context (sequence of token IDs).
        max_new_tokens (int): Maximum number of new tokens to generate.
        context_size (int): Size of the model's context window.
        temperature (float, optional): The temperature for controlling randomness in generation. Defaults to 0.0.
        top_k (int, optional): Limits the sampling pool to the top K tokens with the highest probability. Defaults to None.
        eos_id (int, optional): End-of-sequence token ID. If encountered, generation stops. Defaults to None.

    Returns:
        torch.Tensor: The sequence of token IDs, including newly generated tokens.
    """
    for _ in range(max_new_tokens):
        # Limit the context to the most recent tokens (based on model's context size)
        idx_conds = idx[:, -context_size:]

        # Perform a forward pass without tracking gradients
        with torch.no_grad():  
            logits = model(idx_conds)
            logits = logits[:, -1, :]  # Only take the logits corresponding to the last token

            # Apply top_k filtering if specified, retaining only the top K probabilities
            if top_k is not None:
                max_logits, indices = torch.topk(logits, top_k)
                min_logits = max_logits[:, -1]  # Get the smallest of the top K logits
                logits = torch.where(logits < min_logits, torch.tensor(float('-inf')).to(logits.device), logits)

            # Apply temperature-based sampling if temperature > 0
            if temperature > 0.0:
                logits = logits / temperature
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                # Use greedy sampling (select the token with the highest probability)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Stop generation if end-of-sequence token is encountered
            if next_token == eos_id:
                break

            # Append the generated token to the sequence
            idx = torch.cat([idx, next_token], dim=1)

    return idx


def generate_and_print(model, device, start_context, tokenizer, max_new_tokens=50, temperature=0.0, top_k=25, eos_id=None):
    """
    Generate text from the model and print the decoded result.

    Args:
        model (nn.Module): The language model used for text generation.
        device (torch.device): The device (CPU/GPU) to run the model on.
        start_context (torch.Tensor): Initial context for text generation (sequence of token IDs).
        tokenizer (Tokenizer): Tokenizer to decode the generated tokens into text.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 50.
        temperature (float, optional): The temperature for controlling randomness in generation. Defaults to 0.0.
        top_k (int, optional): Limits the sampling pool to the top K tokens with the highest probability. Defaults to 25.
        eos_id (int, optional): End-of-sequence token ID. If encountered, generation stops. Defaults to None.

    Returns:
        None
    """
    # Set the model to evaluation mode
    model.eval()

    # Get the context size from the model's positional embedding size
    context_size = model.pos_emb.weight.shape[0]

    # Move the start context to the appropriate device (CPU/GPU)
    encoded = start_context.to(device)
    
    # Generate tokens without tracking gradients
    with torch.no_grad():
        token_ids = generate_text(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id
        )
    
    # Decode the generated tokens into text
    decoded_text = token_to_text(token_ids, tokenizer)

    # Print the decoded text (replace newlines with spaces for cleaner output)
    print(decoded_text.replace("\n", " "))

    # Set the model back to training mode
    model.train()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Visualizes the training and validation losses over epochs and the number of tokens seen.

    Args:
        epochs_seen (torch.Tensor): A tensor representing the epochs seen during training.
        tokens_seen (list): A list of tokens seen at each training step.
        train_losses (list): A list of training loss values for each epoch.
        val_losses (list): A list of validation loss values for each epoch.

    Returns:
        None: Displays a plot of the training and validation losses.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training Loss")
    ax1.plot(epochs_seen, val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens Seen")
    
    fig.tight_layout()
    plt.show()


import torch
import os

def save_model(model, optimizer, epoch, file_path):
    """
    Save the model and optimizer state to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        file_path (str): The path where the model will be saved.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the model state
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)
    print(f'Model saved to {file_path}')

def load_model(model, optimizer, file_path):
    """
    Load the model and optimizer state from a file.

    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        file_path (str): The path from which to load the model.
    
    Returns:
        int: The epoch number from the saved state.
    """
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    else:
        print(f"No checkpoint found at {file_path}")
        return 0  # Return 0 if no checkpoint is found
