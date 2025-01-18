import os
import logging
import requests
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)

def download_data(url: str, output_path: str = "instruction_data.json") -> str:
    if not os.path.exists(output_path):
        try:
            logging.info(f"Downloading data from {url}...")
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded data successfully. Saved to {output_path}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed. Error: {e}")
            return None
        except Exception as e:
            logging.error(f"Failed to download data. Error: {e}")
            return None
    else:
        logging.info(f"Data already exists at {output_path}")
    return output_path

def load_data(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return None

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

def split_data(data):
    train_data, temp_data = train_test_split(data, test_size=0.15, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=(10/15), random_state=42)
    logging.info(f"Train shape: {len(train_data)}")
    logging.info(f"Validation shape: {len(val_data)}")
    logging.info(f"Test shape: {len(test_data)}")
    return train_data, val_data, test_data

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_data = [] 
        for entry in data:
            instruction_text = format_input(entry)
            desired_response = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_text + desired_response
            self.encoded_data.append(tokenizer.encode(full_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

def collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_len=None, device="cpu"):
    # Determine the maximum length of the sequences in the batch
    batch_max_length = max(len(item) + 1 for item in batch)
    input_ls, target_ls = [], []

    # Iterate through each item in the batch
    for item in batch:
        # Create a copy of the item and append the padding token
        new_item = item.copy()
        new_item += [pad_token_id]

        # Pad the item to the maximum length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # Separate inputs and targets; inputs exclude the last token, targets exclude the first token
        inputs = torch.tensor(padded[:-1])  # Input tensor
        target = torch.tensor(padded[1:])   # Target tensor

        # Create a mask for the padding tokens in the target tensor
        mask = target == pad_token_id
        indices = torch.nonzero(mask).squeeze()  # Get indices of padding tokens

        # If there are multiple padding tokens, ignore the subsequent ones in the loss calculation
        if indices.numel() > 1:
            target[indices[1:]] = ignore_index  # Set ignore index for padding

        # Optionally limit the length of inputs and targets
        if allowed_max_len is not None:
            inputs = inputs[:allowed_max_len]
            target = target[:allowed_max_len]

        # Append the processed inputs and targets to the respective lists
        input_ls.append(inputs)
        target_ls.append(target)

    # Stack the lists into tensors and move them to the specified device
    input_tensor = torch.stack(input_ls).to(device)
    target_tensor = torch.stack(target_ls).to(device)

    return input_tensor, target_tensor  # Return the batched input and target tensors




device = 'cuda' if torch.cuda.is_available() else 'cpu'
custom_collate_fn = partial(collate_fn, device=device, allowed_max_len=1024)

# Example usage (to be included in the main logic of your application):
if __name__ == "__main__":
    url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

    output_path = download_data(url)
    if output_path:
        data = load_data(output_path)
        if data:
            train_data, val_data, test_data = split_data(data)
            tokenizer = tiktoken.get_encoding('gpt2')

            train_dataset = InstructionDataset(train_data, tokenizer)
            val_dataset = InstructionDataset(val_data, tokenizer)
            test_dataset = InstructionDataset(test_data, tokenizer)

            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
            logging.info("Data loaded successfully.")

            # Do something with the data
            print("Train dataloader:", len(train_loader))
            print("Validation dataloader:", len(val_loader))
            print("Test dataloader:", len(test_loader))


