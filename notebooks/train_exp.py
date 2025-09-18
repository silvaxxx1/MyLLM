#################IMPORTS####################
import os
import gc
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader , Dataset

from UTILS.model import GPTModel
from UTILS.load_weights import download_and_load_gpt2, load_weights_into_gpt 

############################ GPU CLEANUP ####################
def cleanup(model=None, optimizer=None):
    """
    Cleanup function to clear CUDA memory and reset references
    before running a new experiment.
    """
    if model is not None:
        del model
    if optimizer is not None:
        del optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()      # Free up all unused memory
        torch.cuda.synchronize()      # Ensure all pending CUDA ops are done

    gc.collect()
    print("Cleanup done. GPU memory cleared.")

############################CONFIG####################
LOCAL_FRIENDLY_CONFIG = {
    "vocab_size": 50257,        # The total size of the vocabulary (remains unchanged to match tokenizer output)
    "context_length": 256,      # Reduced sequence length from 512 to 256 for smaller memory footprint
    "emb_dim" : 512,            # Reduced embedding dimension from 768 to 512 to lower computational cost
    "n_heads" : 8,              # Reduced the number of attention heads from 12 to 8 to reduce computation per layer
    "n_layers" : 6,             # Reduced the number of layers from 12 to 6 to make the model lighter
    "drop_rate" : 0.1,          # Retained dropout at 10% to regularize training without making it too unstable
    "qkv_bias" : False,         # Keeps the bias configuration for the query, key, and value projections
} 

###################LOAD DATA####################
# Define parameters for DataLoader
batch_size = 8 # Number of samples in each batch. A lower batch size helps fit the data into memory when working with large models.
num_workers = 0  # Disable multiprocessing for data loading, which is recommended when using a CPU only.
pin_memory = True  # This option is useful for GPU training, so we disable it for CPU-only training.

# Dataset class
class GPT2Dataset(Dataset):
    def __init__(self, file_path, max_length, stride):
        # Load the binary file into a NumPy array
        self.data = np.fromfile(file_path, dtype=np.int32)
        self.max_length = max_length  # Maximum length of input sequence
        self.stride = stride          # Stride defines how much to slide the window

    def __len__(self):
        return (len(self.data) - self.max_length) // self.stride

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        input_seq = self.data[start_idx: start_idx + self.max_length]
        output_seq = self.data[start_idx + 1: start_idx + self.max_length + 1]

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(output_seq, dtype=torch.long)
        )


################### SIMULATE DATA (RUNS ONLY ONCE) ####################
data_dir = "./simulated_data"
os.makedirs(data_dir, exist_ok=True)

train_path = os.path.join(data_dir, "train.bin")
val_path = os.path.join(data_dir, "val.bin")
test_path = os.path.join(data_dir, "test.bin")

# Only generate data if it doesn't already exist
if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
    print("Simulated data not found. Generating new data...")

    total_tokens = 5000  # Adjust depending on available memory
    vocab_size = LOCAL_FRIENDLY_CONFIG["vocab_size"]

    # Generate random tokens to simulate tokenized dataset
    data = np.random.randint(0, vocab_size, size=(total_tokens,), dtype=np.int32)

    # Split into train, validation, and test sets
    train_data = data[:3500]
    val_data = data[3500:4200]
    test_data = data[4200:]

    # Save datasets as binary files
    train_data.tofile(train_path)
    val_data.tofile(val_path)
    test_data.tofile(test_path)

    print(f"Simulated data created and saved in: {data_dir}")
else:
    print("Simulated data already exists. Skipping generation.")
print(f"Simulated data saved to: {data_dir}")

################### CREATE DATASETS & LOADERS ####################
train_dataset = GPT2Dataset(train_path, LOCAL_FRIENDLY_CONFIG["context_length"], LOCAL_FRIENDLY_CONFIG["context_length"])
val_dataset = GPT2Dataset(val_path, LOCAL_FRIENDLY_CONFIG["context_length"], LOCAL_FRIENDLY_CONFIG["context_length"])
test_dataset = GPT2Dataset(test_path, LOCAL_FRIENDLY_CONFIG["context_length"], LOCAL_FRIENDLY_CONFIG["context_length"])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers , pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Quick check
for batch in train_loader:
    input_seq, output_seq = batch
    print("Input sequence shape:", input_seq.shape)
    print("Output sequence shape:", output_seq.shape)
    break

################### TRAINING SETUP ####################
# Clear previous memory before starting
cleanup()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(LOCAL_FRIENDLY_CONFIG).to(device)

################### LOSS FUNCTIONS ####################
def cross_entropy_loss_from_scratch(logits, target):
    logits_flatten = logits.flatten(0, 1) 
    target_flatten = target.flatten()
    
    # Stable softmax
    stable_logits = logits_flatten - torch.max(logits_flatten, dim=1, keepdim=True).values
    exp_logits = torch.exp(stable_logits)
    probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
    
    # Pick true class probabilities
    true_probs = torch.gather(probs, dim=1, index=target_flatten.unsqueeze(1)).squeeze(1)
    
    # Negative log-likelihood
    loss = -torch.mean(torch.log(true_probs))
    return loss
    
def calculate_loss_on_batch(model: GPTModel,
                            input_batch: torch.Tensor,
                            target_batch: torch.Tensor,
                            device: torch.device):
    
    # Move data to device
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)

    # Standard PyTorch loss
    loss_batch = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss_batch

def calculate_loss_on_batch_scratch(model: GPTModel,
                            input_batch: torch.Tensor,
                            target_batch: torch.Tensor,
                            device: torch.device):
    
    # Move data to device
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    print("Logits shape:", logits.shape)

    # Standard PyTorch loss
    loss_batch = cross_entropy_loss_from_scratch(logits, target_batch)
    return loss_batch

                               
 # Define a function to calculate the average loss over a specified number of batches from the data loader
def calculate_loss_over_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.

    if len(data_loader) == 0:
        return float("nan")
    
    num_batches = min(num_batches, len(data_loader)) if num_batches is not None else len(data_loader)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # FIX: pass model first
            loss = calculate_loss_on_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

################### EXAMPLE RUN ####################
# input_batch, target_batch = next(iter(train_loader))
# batch_loss = calculate_loss_on_batch(model, input_batch, target_batch, device)
# batch_loss_scratch = calculate_loss_on_batch_scratch(model, input_batch, target_batch, device)
# print(f"Batch loss: {batch_loss}")
# print(f"Batch loss (scratch): {batch_loss_scratch}")

################# fot test : Overfit a single batch ####################
def trainerV0(model, train_loader, optimizer, device, num_epochs=10, use_scratch_loss=False):
    """
    Overfit a single batch to test training loop.
    
    Args:
        model: GPTModel instance
        train_loader: DataLoader for training data
        optimizer: optimizer instance
        device: torch.device
        num_epochs: number of epochs to overfit
        use_scratch_loss: if True, uses cross_entropy_loss_from_scratch
    Returns:
        train_losses: list of loss values per epoch
    """
    
    model.train()  # Ensure model is in training mode
    train_losses = []
    global_step = 0

    # Take a single batch
    input_batch, target_batch = next(iter(train_loader))
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # Slice to context length if needed
    if input_batch.size(1) > model.pos_emb.num_embeddings:
        input_batch = input_batch[:, :model.pos_emb.num_embeddings]
        target_batch = target_batch[:, :model.pos_emb.num_embeddings]

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        logits = model(input_batch)

        # Compute loss
        if use_scratch_loss:
            loss = cross_entropy_loss_from_scratch(logits, target_batch)
        else:
            loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

        # Backprop
        loss.backward()
        optimizer.step()

        global_step += 1
        train_losses.append(loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, Loss: {loss.item():.4f}")

    return train_losses


# Overfit a single batch with PyTorch loss
# print("Overfitting a single batch with PyTorch loss")
# losses_pt = trainerV0(model, train_loader, optimizer, device, num_epochs=50, use_scratch_loss=False)

# Overfit a single batch with your scratch loss
# print("Overfitting a single batch with your scratch loss")
# losses_scratch = trainerV0(model, train_loader, optimizer, device, num_epochs=50, use_scratch_loss=True)

import matplotlib.pyplot as plt
def plot_losses(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', marker='o')  # Plot with markers for clarity
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(len(train_losses)))  # Set x-ticks to be the epoch numbers
    plt.grid()
    plt.legend()
    plt.show()

# plot_losses(losses_pt)
# plot_losses(losses_scratch) 


###################### Trainer V1 : first naive working version #######################


from typing import Optional, Any

import tiktoken 


# helper function to evaluate the model
@torch.no_grad()
def eval_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() 
    train_loss = calculate_loss_over_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss = calculate_loss_over_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss , val_loss
    

## 2- Text Generation 


# A helper function to convert text to token IDs using the tokenizer
def text_to_tokens_ids(text, tokenizer):
    # Encode the input text into token IDs
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})  # Special tokens allowed
    return torch.tensor(encoded).unsqueeze(0)  # Convert to tensor and add batch dimension

# A helper function to decode token IDs back to text using the tokenizer
def token_ids_to_text(ids, tokenizer):
    # Decode the token IDs into a human-readable string
    return tokenizer.decode(ids.squeeze(0).tolist())  # Remove batch dimension and convert to list
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

def generate_and_print_sample(model, tokenizer, device, start_context):
   
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

def TrainerV1(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    eval_iter: int = 10,
    eval_freq: int = 10,
    start_context: Optional[str] = None,
    tokenizer: Optional[Any] = None
):
    
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1  # Track total tokens and global training steps

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode to enable dropout and batch normalization

        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            
            # FIXED: pass model first
            loss = calculate_loss_on_batch(model, input_batch, target_batch, device)
            
            loss.backward()
            optimizer.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = eval_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")


        # Generate and print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen  # Return losses and token tracking information


# Initialize the tokenizer for encoding and decoding text
tok = tiktoken.encoding_for_model("gpt2")



# -------------------- TrainerV2 (updated) --------------------
import os
import time
import warnings

# Silence TensorFlow logs and reduce warnings clutter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def setup_perf_flags(use_tf32: bool = True, cudnn_benchmark: bool = True):
    """Enable/disable TF32 and cudnn benchmark for performance."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(use_tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

def choose_autocast_dtype(preferred: str = "fp16"):
    """Return torch.dtype for autocast, falling back if bf16 not supported."""
    preferred = preferred.lower()
    if preferred == "bf16":
        bf16_ok = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        if bf16_ok:
            return torch.bfloat16
        else:
            warnings.warn("bf16 requested but not supported — falling back to fp16.")
            return torch.float16
    return torch.float16


@torch.no_grad()
def _evaluate_partial(model, data_loader, device, num_batches=5, use_amp=False, autocast_dtype=None):
    """Evaluate for up to num_batches and return average loss."""
    model.eval()
    total = 0.0
    count = 0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch = input_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)

        # crop to model context if available
        if hasattr(model, "pos_emb"):
            ctx_len = model.pos_emb.num_embeddings
            if input_batch.size(1) > ctx_len:
                input_batch = input_batch[:, :ctx_len]
                target_batch = target_batch[:, :ctx_len]

        # use new amp API
        if use_amp and device.type == "cuda":
            autocast_ctx = torch.amp.autocast("cuda", enabled=True, dtype=autocast_dtype)
        else:
            autocast_ctx = torch.amp.autocast("cpu", enabled=False)

        with autocast_ctx:
            logits = model(input_batch)  # [B, T, V]
            loss = F.cross_entropy(logits.flatten(0,1), target_batch.flatten())
        total += float(loss.item())
        count += 1
    model.train()
    return total / max(1, count)

import datetime

def log_info(message: str, level: str = "INFO"):
    """Simple timestamped logging for better readability."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {message}")


def TrainerV2(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 1,
    *,
    use_amp: bool = True,
    amp_dtype: str = "fp16",
    use_compile: bool = True,
    use_tf32: bool = True,  # ✅ Added TensorFloat32 toggle
    accumulate_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    scheduler=None,
    eval_freq_steps: int = 200,
    eval_iter: int = 5,
    start_context: str | None = None,
    tokenizer=None,
    print_every: int = 50
):

    log_info("Starting training session", "INIT")

    # AMP setup
    use_cuda = (device.type == "cuda")
    autocast_dtype = choose_autocast_dtype(amp_dtype) if use_amp and use_cuda else None

    # Compile model
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune", backend="inductor")
            log_info("Model compiled with torch.compile (mode=max-autotune)", "MODEL")
        except Exception as e:
            log_info(f"torch.compile failed: {e} — continuing without compile", "WARN")

    model.to(device)

    # GradScaler for AMP
    scaler = torch.amp.GradScaler(
        device="cuda" if use_amp and use_cuda else None,
        enabled=use_amp and use_cuda
    )

    train_log, val_log = [], []
    global_step, tokens_seen = 0, 0
    t0 = time.time()

    for epoch in range(num_epochs):
        log_info(f"Epoch {epoch+1} started", "EPOCH")
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for step, (input_batch, target_batch) in enumerate(train_loader):
            # Move data to device
            input_batch = input_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            # Context crop
            ctx_len = getattr(model, "pos_emb", None)
            if ctx_len is not None:
                ctx_len = ctx_len.num_embeddings
                input_batch = input_batch[:, :ctx_len]
                target_batch = target_batch[:, :ctx_len]

            # Mixed precision context
            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=autocast_dtype)
                if use_amp and use_cuda
                else torch.amp.autocast("cpu", enabled=False)
            )

            with autocast_ctx:
                logits = model(input_batch)
                loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten()) / accumulate_steps

            # Backward
            if use_amp and use_cuda:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if (step + 1) % accumulate_steps == 0:
                if use_amp and use_cuda:
                    scaler.unscale_(optimizer)
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if use_amp and use_cuda:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()

            # Update counters
            step_loss_val = float(loss.item() * accumulate_steps)
            epoch_loss_sum += step_loss_val
            epoch_steps += 1
            global_step += 1
            tokens_seen += input_batch.numel()

            # Log step progress
            if global_step % print_every == 0:
                elapsed = time.time() - t0
                log_info(
                    f"Epoch {epoch+1} | Step {global_step} | Loss: {step_loss_val:.4f} | "
                    f"Tokens Seen: {tokens_seen:,} | Elapsed: {elapsed:.1f}s",
                    "STEP"
                )

            # Periodic eval
            if eval_freq_steps and global_step % eval_freq_steps == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = _evaluate_partial(model, train_loader, device, eval_iter, use_amp, autocast_dtype)
                    val_loss = _evaluate_partial(model, val_loader, device, eval_iter, use_amp, autocast_dtype)
                train_log.append({"step": global_step, "epoch": epoch+1, "train_loss": train_loss})
                val_log.append({"step": global_step, "epoch": epoch+1, "val_loss": val_loss})
                log_info(f"Eval — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", "EVAL")
                model.train()

        # Epoch end
        avg_loss = epoch_loss_sum / max(1, epoch_steps)
        log_info(f"Epoch {epoch+1} finished — Avg Loss: {avg_loss:.4f}", "EPOCH END")

        # Optional sample generation
        if start_context and tokenizer:
            try:
                log_info("Generated Sample:", "GEN")
                generate_and_print_sample(model, tokenizer, device, start_context)
            except Exception as e:
                log_info(f"Sample generation failed: {e}", "WARN")

    return train_log, val_log, tokens_seen



# -------------------- example call (unchanged) --------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_log, val_log, tokens_seen = TrainerV2(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=3,
    use_amp=True,
    amp_dtype="fp16",      # or "bf16" if your hardware & PyTorch support it
    use_compile=True,
    use_tf32=True,
    accumulate_steps=1,
    max_grad_norm=1.0,
    scheduler=scheduler,
    eval_freq_steps=200,
    eval_iter=4,
    start_context="Once upon a time",
    tokenizer=tok,
    print_every=10
)
