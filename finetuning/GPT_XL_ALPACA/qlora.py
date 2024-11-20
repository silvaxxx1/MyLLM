import os
import argparse
import time
import logging
import torch
from torch.utils.data import DataLoader
import torch._dynamo
from data import InstructionDataset, load_data, split_data, download_data, custom_collate_fn
import tiktoken
from model import GPTModel
from qlora_model import replace_with_qlora
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    # Suppress Dynamo errors
    torch._dynamo.config.suppress_errors = True

    # Download and load data
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
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

    def load_lora_model(model_path, model_config, rank=8, num_bits=4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LoRA model file not found at {model_path}")

        # Load base GPT2 model first
        model = GPTModel(BASE_CONFIG)

        # Inject LoRA layers into the model
        model = replace_with_qlora(model, rank=rank, num_bits=num_bits)

        # Load the pre-trained weights into the model
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        logging.info(f"LoRA-enhanced model loaded from {model_path}")
        return model


    # Load the model
    model_path = "./lora_models/gpt2_lora.pth"
    model = load_lora_model(model_path, BASE_CONFIG, rank=args.rank, num_bits=args.num_bits)    

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/fine_tuning')

    # Initialize and train the model
    fine_tuner = FineTuner(model=model, train_loader=train_loader, val_loader=val_loader, 
                           num_epochs=args.num_epochs, learning_rate=args.learning_rate, writer=writer)

    # Start the training process
    fine_tuner.train_model()

class FineTuner:
    def __init__(self, model, train_loader, val_loader, num_epochs=5, learning_rate=5e-5, device=None, writer=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device if device else torch.device('cpu')
        self.writer = writer

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.model.to(self.device)

    def train_model(self):
        """Train the model for the specified number of epochs."""
        self.model.train()
        best_val_loss = float('inf')
        early_stopping_counter = 0

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

            # Write training metrics to TensorBoard
            self.writer.add_scalar('Loss/train', average_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_accuracy * 100, epoch)

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

            # Write validation metrics to TensorBoard
            self.writer.add_scalar('Loss/val', average_val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_accuracy * 100, epoch)

            # Save model checkpoint if validation loss improves
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                early_stopping_counter = 0
                self.save_checkpoint(epoch)
            else:
                early_stopping_counter += 1

            # Early stopping check
            if early_stopping_counter >= 3:  # Stop if no improvement for 3 epochs
                logging.info("Early stopping triggered. Validation loss has not improved.")
                break

            # Estimate and log time remaining
            epoch_duration = time.time() - start_time
            remaining_time = epoch_duration * (self.num_epochs - epoch - 1)
            logging.info(f"Time per epoch: {epoch_duration:.2f}s. Estimated time remaining: {remaining_time/60:.2f} minutes.")

    def save_checkpoint(self, epoch):
        checkpoint_dir = './checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(self.model.state_dict(), checkpoint_path)
        logging.info(f"Checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    # Argument parser for command line options
    parser = argparse.ArgumentParser(description='Fine-tune GPT model.')
    parser.add_argument('--model', type=str, default='gpt2-medium (355M)', 
                        choices=['gpt2-small (124M)', 'gpt2-medium (355M)', 'gpt2-large (774M)', 'gpt2-xl (1558M)'],
                        help='Choose the model architecture to use.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to train the model.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for optimization.')
    parser.add_argument('--rank', type=int, default=8, help='Rank of LoRA layers.')
    parser.add_argument('--num_bits', type=int, default=4, help='Number of bits for quantization.')

    args = parser.parse_args()
    main(args)
