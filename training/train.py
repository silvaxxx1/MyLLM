import torch
import torch.nn as nn
from torch.nn import functional as F
from train_utils import generate_and_print

def loss_batch(input_batch, target_batch, model, device):
    """
    Calculate the loss for a single batch of input and target.

    Args:
        input_batch (torch.Tensor): Input data batch.
        target_batch (torch.Tensor): Corresponding target data batch.
        model (nn.Module): The model to train.
        device (torch.device): Device to run the model on.

    Returns:
        torch.Tensor: The calculated loss for the given batch.
    """
    # Move data to the device (CPU/GPU)
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    
    # Forward pass: Get model predictions (logits)
    logits = model(input_batch)
    
    # Calculate the cross-entropy loss (flatten the logits and target for batch processing)
    loss_batch = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    return loss_batch


def loss_fn(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss over a set of batches from the data loader.

    Args:
        data_loader (DataLoader): The data loader providing batches of input/target.
        model (nn.Module): The model to evaluate.
        device (torch.device): Device to run the model on.
        num_batches (int, optional): Number of batches to evaluate. Defaults to the full dataset.

    Returns:
        float: The average loss over the evaluated batches.
    """
    total_loss = 0

    # Handle case where data loader is empty
    if len(data_loader) == 0:
        return float("nan")
    
    # If num_batches is not provided, evaluate over the entire data loader
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)

    # Iterate through the data loader and compute the loss for each batch
    for i, (batch, target) in enumerate(data_loader):
        if i < num_batches:
            loss = loss_batch(batch, target, model, device)
            total_loss += loss
        else:
            break

    # Return the average loss over the evaluated batches
    return total_loss / num_batches 


def trainV1(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """
    Training loop for the model with periodic evaluation and generation.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        device (torch.device): Device to run the model on.
        num_epochs (int): Number of training epochs.
        eval_freq (int): Frequency (in steps) to evaluate and generate text.
        eval_iter (int): Number of batches to evaluate during evaluation.
        start_context (torch.Tensor): Initial context for text generation.
        tokenizer (Tokenizer): Tokenizer to decode generated text.

    Returns:
        tuple: Training losses, validation losses, and token counts seen over the course of training.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1  # Initialize counters for tokens and global step

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        for input_batch, target_batch in train_loader:
            # Zero gradients before the backward pass
            optimizer.zero_grad()
            
            # Compute loss for the current batch
            loss = loss_fn(train_loader, model, device)
            
            # Backpropagate the loss
            loss.backward()
            
            # Update model parameters
            optimizer.step()

            # Track the number of tokens seen and the global training step
            token_seen += input_batch.shape[0]
            global_step += 1 

            # Perform evaluation and text generation periodically
            if global_step % eval_freq == 0:
                # Evaluate the model on training and validation sets
                train_loss, val_loss = eval_model(
                    model, train_loader, val_loader, 
                    device, eval_iter
                )
                # Log the losses and tokens seen
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)

                # Print the progress
                print(f"Epoch {epoch + 1} | Step {global_step} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f}")

                # Generate and print text samples from the model
                generate_and_print(model, device, start_context, tokenizer)

    # Return training/validation losses and tokens seen after the complete training
    return train_losses, val_losses, track_tokens_seen



import math
import torch

def trainV2(model, train_loader, val_loader, optimizer, device,
            n_epochs, eval_freq, eval_iter, start_context, tokenizer,
            warmup_steps, initial_lr=3e-05, min_lr=1e-6):
    """
    Train a model with a more advanced training loop that includes learning rate scheduling,
    gradient clipping, and periodic evaluation.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        device (torch.device): The device to run the training on (CPU or GPU).
        n_epochs (int): The number of epochs to train the model.
        eval_freq (int): The frequency of evaluations (in steps).
        eval_iter (int): The number of batches to evaluate.
        start_context (str): The initial context for generating samples.
        tokenizer: The tokenizer used for the model.
        warmup_steps (int): The number of steps for the learning rate warmup phase.
        initial_lr (float): The initial learning rate during the warmup.
        min_lr (float): The minimum learning rate during cosine annealing.

    Returns:
        tuple: Contains lists of training losses, validation losses, tokens seen,
               and learning rates during training.
    """
    # Lists to track training and validation losses, tokens seen, and learning rates
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # Retrieve the maximum learning rate from the optimizer
    peak_lr = optimizer.param_groups[0]["lr"]

    # Calculate the total number of training steps
    total_training_steps = len(train_loader) * n_epochs

    # Calculate the learning rate increment during the warmup phase
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        model.train()  # Set the model to training mode
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset gradients
            global_step += 1  # Increment global step

            # Adjust the learning rate based on the current phase (warmup or cosine annealing)
            if global_step < warmup_steps:
                # Linear warmup
                lr = initial_lr + global_step * lr_increment  
            else:
                # Cosine annealing after warmup
                progress = ((global_step - warmup_steps) / 
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # Store the current learning rate

            # Calculate and backpropagate the loss
            loss = loss_fn(input_batch, target_batch, model, device)
            loss.backward()  # Backpropagation

            # Apply gradient clipping after the warmup phase to avoid exploding gradients
            if global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()  # Update model parameters
            tokens_seen += input_batch.numel()  # Count the number of tokens seen

            # Periodically evaluate the model on the training and validation sets
            if global_step % eval_freq == 0:
                train_loss, val_loss = eval_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # Print the current losses
                print(f"Ep {epoch + 1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        # Generate and print a sample from the model to monitor progress
        generate_and_print(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen, track_lrs



def eval_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate the model on the training and validation datasets.

    Args:
        model (nn.Module): The model to evaluate.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model on.
        eval_iter (int): Number of batches to evaluate.

    Returns:
        tuple: Average training loss and validation loss.
    """
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # Disable gradient calculations
        # Compute average loss for the training and validation datasets
        train_loss = loss_fn(train_loader, model, device, num_batches=eval_iter)
        val_loss = loss_fn(val_loader, model, device, num_batches=eval_iter)
    
    model.train()  # Switch back to training mode
    return train_loss, val_loss

