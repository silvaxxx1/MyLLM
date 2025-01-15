import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Define a function to calculate the average loss over all batches from the data loader

# Define a function to calculate the loss for a single batch
def calc_loss_batch(input_batch, target_batch, model, device):
    
    # Move input and target batches to the appropriate device (CPU/GPU)
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # Forward pass: Get model predictions (logits) for the input batch
    logits = model(input_batch)

    # Calculate the cross-entropy loss by flattening logits and targets
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    
    return loss  # Return the computed loss


# Define a function to calculate the average loss over a specified number of batches from the data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    
    total_loss = 0.  # Initialize total loss to zero

    # Check if the data loader is empty and return NaN if it is
    if len(data_loader) == 0:
        return float("nan")
    
    # Determine the number of batches to process
    elif num_batches is None:
        num_batches = len(data_loader)  # Use all batches
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))

    # Iterate through the batches in the data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:  # Process only the specified number of batches
            # Calculate loss for the current batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # Accumulate the loss
        else:
            break  # Exit if the specified number of batches has been processed

    # Return the average loss across the specified number of batches
    return total_loss / num_batches

# A helper function to generate text during the training process
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is a (B, T) array of indices representing the current context
    # B: batch size, T: current sequence length

    for _ in range(max_new_tokens):
        # Loop to generate tokens one at a time until the desired number is reached

        # Crop the current context if it exceeds the supported context size
        # For instance, if the model supports a maximum context size of 5 tokens,
        # but we provide a sequence of 10 tokens, we only use the last 5 tokens.
        idx_cond = idx[:, -context_size:]  # Use only the last context_size tokens

        # Get the model's predictions for the current context
        with torch.no_grad():  # Disable gradient calculation for efficiency during inference
            logits = model(idx_cond)  # Feed the context into the model to get predictions

        # Focus only on the last time step's logits
        # This reduces the logits shape from (batch, n_tokens, vocab_size) to (batch, vocab_size)
        logits = logits[:, -1, :]

        # Determine the index of the vocabulary entry with the highest logit value (greedy sampling)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append the sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens + 1)

    return idx  # Return the final sequence of indices with the newly generated tokens

# A helper function to convert text to token IDs using the tokenizer
def text_to_tokens_ids(text, tokenizer):
    # Encode the input text into token IDs
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})  # Special tokens allowed
    return torch.tensor(encoded).unsqueeze(0)  # Convert to tensor and add batch dimension

# A helper function to decode token IDs back to text using the tokenizer
def token_ids_to_text(ids, tokenizer):
    # Decode the token IDs into a human-readable string
    return tokenizer.decode(ids.squeeze(0).tolist())  # Remove batch dimension and convert to list


## 1- Model Evaluation 

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluates the model's performance on the training and validation datasets.

    Args:
        model: The model to evaluate.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        device: The device (CPU or GPU) to perform computations on.
        eval_iter: Number of batches to use for evaluation.

    Returns:
        train_loss: Average loss on the training dataset.
        val_loss: Average loss on the validation dataset.
    """
    model.eval()  # Set the model to evaluation mode to disable dropout and batch normalization
    with torch.no_grad():  # Disable gradient tracking for efficiency during evaluation
        # Calculate the average loss for the training dataset
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        # Calculate the average loss for the validation dataset
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # Set the model back to training mode
    return train_loss, val_loss  # Return the training and validation losses


## 2- Text Generation 

def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generates and prints a sample of text based on the provided starting context.

    Args:
        model: The model used for text generation.
        tokenizer: The tokenizer used to encode and decode text.
        device: The device (CPU or GPU) for computation.
        start_context: The initial text input for generating new tokens.
    """
    model.eval()  # Set the model to evaluation mode to disable dropout and batch normalization
    context_size = model.pos_emb.weight.shape[0]  # Determine the context size from the model's positional embeddings
    # Encode the starting context into token IDs and move to the appropriate device
    encoded = text_to_tokens_ids(start_context, tokenizer).to(device)
    with torch.no_grad():  # Disable gradient tracking during text generation
        # Generate new token IDs using the helper function
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    # Decode the generated token IDs back to text
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    # Print the generated text, replacing newlines for compact formatting
    print(decoded_text.replace("\n", " "))  
    model.train()  # Set the model back to training mode


def trainerV1(model, train_loader, val_loader, optimizer, device, num_epochs,
               eval_freq, eval_iter, start_context, tokenizer):
    """
    Trains the model over a specified number of epochs, evaluating its performance at regular intervals.

    Args:
        model: The model to be trained.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        optimizer: The optimizer for updating model parameters.
        device: The device (CPU or GPU) to perform computations on.
        num_epochs: Total number of training epochs.
        eval_freq: Frequency of evaluation (in steps).
        eval_iter: Number of batches to use for evaluation.
        start_context: Initial context for text generation.
        tokenizer: Tokenizer used to encode and decode text.

    Returns:
        train_losses: List of training losses over the epochs.
        val_losses: List of validation losses over the epochs.
        track_tokens_seen: List of total tokens seen during training.
    """
    
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1  # Track total tokens and global training steps

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode to enable dropout and batch normalization

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # Calculate the loss
            loss.backward()  # Backpropagate to calculate gradients
            optimizer.step()  # Update model weights using the calculated gradients
            
            tokens_seen += input_batch.numel()  # Count the total number of tokens processed
            global_step += 1  # Increment the global step counter

            # Optional evaluation step
            if global_step % eval_freq == 0:  # Evaluate the model at specified intervals
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)  # Store training loss
                val_losses.append(val_loss)  # Store validation loss
                track_tokens_seen.append(tokens_seen)  # Store total tokens seen
                
                # Print training and validation loss information
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Generate and print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen  # Return losses and token tracking information


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label=f"Training {label}")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    plt.show()