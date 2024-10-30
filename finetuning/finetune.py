import torch
from torch.utils.data import DataLoader, Dataset
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import pandas as pd
import tiktoken
from utility import download_and_load_gpt2, load_weights_into_gpt
from model import GPTModel
import time
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)  # Load data from the CSV file
        logging.info(f'Dataset loaded from {csv_file} with {len(self.data)} samples.')

        # Pre-tokenize texts using the provided tokenizer
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["message"]
        ]

        # Determine the maximum length for encoding
        if max_length is None:
            self.max_length = self._longest_encoded_length()  # Set max length to the longest encoded length
            logging.info(f'Max length not provided. Using {self.max_length} as max length.')
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
            logging.info(f'Max length provided: {self.max_length}. Truncating sequences.')

        # Pad sequences to ensure they all have the same length
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]  # Get the encoded text at the specified index
        label = self.data.iloc[index]["label"]  # Get the corresponding label
        return (
            torch.tensor(encoded, dtype=torch.long),  # Convert encoded text to a tensor
            torch.tensor(label, dtype=torch.long)     # Convert label to a tensor
        )

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)  # Return the length of the dataset

    def _longest_encoded_length(self):
        max_length = 0  # Initialize max_length to 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)  # Get the length of the current encoded text
            if encoded_length > max_length:
                max_length = encoded_length  # Update max_length if the current length is greater
        return max_length  # Return the longest length found

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fine-tune a GPT model on spam detection.")
parser.add_argument("--train_data", type=str, default="finetune_data/train_data.csv",
                    help="Path to the training data CSV file.")
parser.add_argument("--val_data", type=str, default="finetune_data/validation_data.csv",
                    help="Path to the validation data CSV file.")
parser.add_argument("--test_data", type=str, default="finetune_data/test_data.csv",
                    help="Path to the test data CSV file.")
parser.add_argument("--num_epochs", type=int, default=5,
                    help="Number of epochs for training.")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size for training.")
parser.add_argument("--learning_rate", type=float, default=5e-5,
                    help="Learning rate for the optimizer.")
parser.add_argument("--max_length", type=int, default=None,
                    help="Maximum length of input sequences.")
args = parser.parse_args()

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load datasets
train_dataset = SpamDataset(csv_file=args.train_data, tokenizer=tokenizer, max_length=args.max_length)
test_dataset = SpamDataset(csv_file=args.test_data, tokenizer=tokenizer, max_length=args.max_length)
val_dataset = SpamDataset(csv_file=args.val_data, tokenizer=tokenizer, max_length=args.max_length)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Choose the model to use
CHOOSE_MODEL = "gpt2-small (124M)"

# Base configuration settings for the model
BASE_CONFIG = {
    "vocab_size": 50257,     # Size of the vocabulary used by the model
    "context_length": 1024,  # Maximum context length the model can handle
    "drop_rate": 0.0,        # Dropout rate for regularization
    "qkv_bias": True         # Whether to use bias terms in query, key, and value projections
}

# Dictionary containing configurations for different GPT-2 model sizes
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},      # Config for small model
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},    # Config for medium model
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},     # Config for large model
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},       # Config for extra-large model
}

# Update the BASE_CONFIG with parameters specific to the chosen model
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Assert to check that the maximum length of the training dataset does not exceed the model's context length
assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)

logging.info('Model configuration completed.')

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# training loop (finetune)
model = torch.compile(model)

class FineTuner:
    def __init__(self, model, train_loader, val_loader, num_epochs=5, learning_rate=5e-5, device=None):
        self.model = model  # Store the model
        self.train_loader = train_loader  # Store the training DataLoader
        self.val_loader = val_loader  # Store the validation DataLoader
        self.num_epochs = num_epochs  # Store the number of epochs
        self.device = device if device else torch.device('cpu')  # Set the device to GPU or CPU

        # Initialize the optimizer with the provided learning rate
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.model.to(self.device)  # Move the model to the specified device

    def train_model(self):
        """Train the model for the specified number of epochs."""
        self.model.train()  # Set the model to training mode
        for epoch in range(self.num_epochs):
            start_time = time.time()  # Start timing the epoch
            total_loss = 0  # Initialize total loss for the epoch
            correct_predictions = 0  # To track correct predictions during the epoch

            logging.info(f'Starting epoch {epoch + 1}/{self.num_epochs}')

            for step, batch in enumerate(self.train_loader):
                inputs, labels = batch  # Unpack the input data and labels
                inputs = inputs.to(self.device)  # Move inputs to the specified device
                labels = labels.to(self.device)  # Move labels to the specified device

                self.optimizer.zero_grad()  # Reset gradients before backward pass

                outputs = self.model(inputs)  # Forward pass through the model
                logits = outputs[:, -1, :]  # Get the logits for the last output token
                loss = torch.nn.functional.cross_entropy(logits, labels)  # Calculate the loss

                loss.backward()  # Backward pass to calculate gradients
                self.optimizer.step()  # Update the model weights

                total_loss += loss.item()  # Accumulate loss

                # Track correct predictions
                _, preds = torch.max(logits, dim=1)  # Get predicted labels
                correct_predictions += torch.sum(preds == labels).item()  # Count correct predictions

                # Log metrics for the current step
                step_accuracy = correct_predictions / ((step + 1) * self.train_loader.batch_size)
                if (step + 1) % 10 == 0:  # Log every 10 steps
                    logging.info(f"Step {step + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}, "
                                 f"Accuracy: {step_accuracy * 100:.2f}%")

            average_loss = total_loss / len(self.train_loader)  # Calculate average loss for the epoch
            train_accuracy = correct_predictions / len(self.train_loader.dataset)  # Calculate training accuracy

            # Log metrics at the end of the epoch
            logging.info(f"Epoch {epoch + 1} completed. Average Loss: {average_loss:.4f}, "
                         f"Training Accuracy: {train_accuracy * 100:.2f}%")

            # Evaluate the model on the validation set after each epoch
            val_loss, val_accuracy = self.evaluate_model()  # Call the evaluate_model method
            logging.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

    def evaluate_model(self):
        """Evaluate the model on the validation set."""
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0  # Initialize total loss for validation
        correct_predictions = 0  # To track correct predictions during validation

        with torch.no_grad():  # Disable gradient calculation
            for batch in self.val_loader:
                inputs, labels = batch  # Unpack the input data and labels
                inputs = inputs.to(self.device)  # Move inputs to the specified device
                labels = labels.to(self.device)  # Move labels to the specified device

                outputs = self.model(inputs)  # Forward pass through the model
                logits = outputs[:, -1, :]  # Get the logits for the last output token
                loss = torch.nn.functional.cross_entropy(logits, labels)  # Calculate the loss

                total_loss += loss.item()  # Accumulate loss

                # Track correct predictions
                _, preds = torch.max(logits, dim=1)  # Get predicted labels
                correct_predictions += torch.sum(preds == labels).item()  # Count correct predictions

        average_loss = total_loss / len(self.val_loader)  # Calculate average loss for validation
        val_accuracy = correct_predictions / len(self.val_loader.dataset)  # Calculate validation accuracy

        return average_loss, val_accuracy  # Return the average loss and accuracy

# Initialize the FineTuner class with the model and data loaders using command-line arguments
fine_tuner = FineTuner(model=model, train_loader=train_dataloader, val_loader=val_dataloader, 
                       num_epochs=args.num_epochs, learning_rate=args.learning_rate)

# Start the training process
fine_tuner.train_model()
