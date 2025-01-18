# test_dataloader.py
from dataloader import GPTDataLoader

# Specify your parameters
path_file = './train_ids.bin'  # Adjust this to your actual path
max_len = 32
stride = 16
batch_size = 8

# Create the data loader
train_loader = GPTDataLoader(path_file, max_len, stride, batch_size)

# Test loading a batch
for input_batch, target_batch in train_loader:
    print("Input batch:", input_batch)
    print("Target batch:", target_batch)
    break  # Just test the first batch
