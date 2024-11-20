import argparse
import time
import logging
import torch
from data import *
from torch.utils.data import DataLoader
import torch._dynamo
from data import InstructionDataset, load_data, split_data, download_data, custom_collate_fn
from utils import download_and_load_gpt2, load_weights_into_gpt
from model import GPTModel
import tiktoken

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    # Suppress Dynamo errors
    torch._dynamo.config.suppress_errors = True

    # Download and load data
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    output_path = download_data(url)
    data = load_data(output_path)
    train_data, val_data, test_data = split_data(data)

    # Tokenizer initialization
    tokenizer = tiktoken.get_encoding('gpt2')
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    # Load datasets into DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, drop_last=True)   
    logging.info("Data loaded successfully.")

    # Model configuration
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    
    # Update the BASE_CONFIG with parameters specific to the chosen model
    BASE_CONFIG.update(model_configs[args.model])
    logging.info('Model configuration completed.')

    # Load the model
    model_size = args.model.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    model = torch.compile(model)

    # Initialize and train the model
    fine_tuner = FineTuner(model=model, train_loader=train_loader, val_loader=val_loader, 
                           num_epochs=args.num_epochs, learning_rate=args.learning_rate)

    # Start the training process
    fine_tuner.train_model()

class FineTuner:
    def __init__(self, model, train_loader, val_loader, num_epochs=5, learning_rate=5e-5, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device if device else torch.device('cpu')

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.model.to(self.device)

    def train_model(self):
        """Train the model for the specified number of epochs."""
        self.model.train()
        for epoch in range(self.num_epochs):
            start_time = time.time()
            total_loss = 0
            correct_predictions = 0

            logging.info(f'Starting epoch {epoch + 1}/{self.num_epochs}')

            # Training loop
            for step, batch in enumerate(self.train_loader):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                logits = outputs
                loss = torch.nn.functional.cross_entropy(logits, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels).item()

                step_accuracy = correct_predictions / ((step + 1) * self.train_loader.batch_size)
                if (step + 1) % 10 == 0:
                    logging.info(f"Step {step + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}, "
                                 f"Accuracy: {step_accuracy * 100:.2f}%")

            average_loss = total_loss / len(self.train_loader)
            train_accuracy = correct_predictions / len(self.train_loader.dataset)

            # Log training metrics
            logging.info(f"Epoch {epoch + 1} completed. Average Loss: {average_loss:.4f}, "
                         f"Training Accuracy: {train_accuracy * 100:.2f}%")

            # Validation step
            self.model.eval()  # Set the model to evaluation mode
            val_loss = 0
            val_correct_predictions = 0

            with torch.no_grad():  # Disable gradient computation for validation
                for batch in self.val_loader:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    logits = outputs
                    loss = torch.nn.functional.cross_entropy(logits, labels)

                    val_loss += loss.item()
                    _, preds = torch.max(logits, dim=1)
                    val_correct_predictions += torch.sum(preds == labels).item()

            average_val_loss = val_loss / len(self.val_loader)
            val_accuracy = val_correct_predictions / len(self.val_loader.dataset)

            # Log validation metrics
            logging.info(f"Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Argument parser for command line options
    parser = argparse.ArgumentParser(description='Fine-tune GPT model.')
    parser.add_argument('--model', type=str, default='gpt2-medium (355M)', 
                        choices=['gpt2-small (124M)', 'gpt2-medium (355M)', 'gpt2-large (774M)', 'gpt2-xl (1558M)'],
                        help='Choose the model to fine-tune')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    
    args = parser.parse_args()
    
    main(args)
