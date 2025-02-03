import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    """
    A custom dataset class for preparing data for training a GPT-like model.
    
    Args:
        path_file (str): Path to the binary input data file.
        max_len (int): Maximum length of each input sequence.
        stride (int): Stride length for creating overlapping sequences.
    """
    
    def __init__(self, path_file, max_len, stride):
        self.input_ids = []
        self.target_ids = []
        self.data = self.load_data(path_file)

        # Create input-target pairs using sliding windows
        for i in range(0, len(self.data) - max_len, stride):
            input_block = self.data[i: i + max_len]
            target_block = self.data[i + 1: i + max_len + 1]
            self.input_ids.append(torch.tensor(input_block, dtype=torch.long))  # Ensure LongTensor for model
            self.target_ids.append(torch.tensor(target_block, dtype=torch.long))

    def load_data(self, path_file):
        """Load data from a binary file and return as a numpy array."""
        return np.fromfile(path_file, dtype=np.uint16)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Return the input and target tensors for the given index."""
        return self.input_ids[idx], self.target_ids[idx]

def GPTDataLoader(path_file, max_len, stride, batch_size):
    """Create a DataLoader for the GPTDataset."""
    dataset = GPTDataset(path_file, max_len, stride)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=0)
    return dataloader

# Usage example
if __name__ == "__main__":
    # Example parameters
    path_file = 'train_ids.bin'  # Change to your train data file
    max_len = 64  # Example maximum sequence length
    stride = 32  # Example stride
    batch_size = 32  # Example batch size

    # Create DataLoader
    train_loader = GPTDataLoader(path_file, max_len, stride, batch_size)

    # Iterate through DataLoader
    for input_batch, target_batch in train_loader:
        print("Input batch:", input_batch)
        print("Target batch:", target_batch)
        break  # Print only the first batch for demonstration
