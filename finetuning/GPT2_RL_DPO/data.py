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

class PrefrenceDataset(Dataset):
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

### collcate function here

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu"
):
    # Step 1: Initialize a dictionary to hold batch data
    batch_data = {
        "prompt": [],        # Stores tokenized prompts
        "chosen": [],        # Stores tokenized and padded chosen responses
        "rejected": [],      # Stores tokenized and padded rejected responses
        "rejected_mask": [], # Mask indicating valid tokens in rejected responses
        "chosen_mask": []    # Mask indicating valid tokens in chosen responses
    }

    # Step 2: Determine the maximum sequence length across the batch
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            # Find the longest sequence length for the given key
            current_max = max(len(item[key]) + 1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # Step 3: Process each item in the batch
    for item in batch:
        # Step 3.1: Convert the prompt tokens to a tensor and store
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            # Step 3.2: Pad the sequence to the maximum common length
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))

            # Step 3.3: Create a mask where valid tokens are True and padding is False
            mask = torch.ones(len(padded)).bool()
            mask[len(sequence):] = False  # Set padding tokens to False

            # Step 3.4: Optionally mask prompt tokens in the sequence
            if mask_prompt_tokens:
                mask[:prompt.shape[0] + 2] = False  # Exclude prompt and "### Response"

            # Step 3.5: Append the padded sequence and mask to batch data
            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # Step 4: Stack sequences and masks into tensors and finalize
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        # Step 4.1: Stack all sequences for the given key into a tensor
        tensor_stack = torch.stack(batch_data[key])

        # Step 4.2: Optionally truncate tensors to the allowed maximum length
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # Step 4.3: Move tensors to the specified device
        batch_data[key] = tensor_stack.to(device)

    # Step 5: Return the processed batch data
    return batch_data


device = 'cuda' if torch.cuda.is_available() else 'cpu'
custom_collate_fn = partial(custom_collate_fn, device=device, allowed_max_len=1024)

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

            train_dataset = PrefrenceDataset(train_data, tokenizer)
            val_dataset = PrefrenceDataset(val_data, tokenizer)
            test_dataset = PrefrenceDataset(test_data, tokenizer)

            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
            logging.info("Data loaded successfully.")

            # Do something with the data
            print("Train dataloader:", len(train_loader))
            print("Validation dataloader:", len(val_loader))
            print("Test dataloader:", len(test_loader))


