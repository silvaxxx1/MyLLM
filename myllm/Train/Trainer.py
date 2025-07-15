import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Calculate loss for a single batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# Calculate average loss over batches from data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches

# Updated trainerV0: now accepts cfg explicitly and uses it for context_length
def trainerV0(model, train_loader, optimizer, device, num_epochs, cfg):
    model.to(device)
    train_losses = []
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for input_batch, target_batch in train_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            # Clip input length if needed
            if input_batch.size(1) > cfg.block_size:
                input_batch = input_batch[:, :cfg.block_size]
                target_batch = target_batch[:, :cfg.block_size]

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_batches += 1
            global_step += 1

            print(f"Epoch {epoch + 1}, Step {global_step}, Batch {num_batches}, Loss: {batch_loss:.4f}")

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

    return train_losses



# Import your GPT and ModelConfig classes
from model import GPT
from ModelConfig import ModelConfig

cfg = ModelConfig.from_name("gpt2-small")
model = GPT(cfg)

# Dataset and DataLoader setup
from torch.utils.data import Dataset, DataLoader

class RandomTextDataset(Dataset):
    def __init__(self, vocab_size=50257, block_size=1024, num_samples=1000):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(low=0, high=self.vocab_size, size=(self.block_size,), dtype=torch.long)
        y = x.clone()
        return x, y

dataset = RandomTextDataset()
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 3

# Call trainerV0 with cfg explicitly passed in
trainerV0(model, train_loader, optimizer, device, num_epochs, cfg)
