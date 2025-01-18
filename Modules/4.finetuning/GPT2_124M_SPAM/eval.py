import argparse
import torch
import matplotlib.pyplot as plt
from model import GPTModel  # Import your model class here
from gpt2_spam_finetune import SpamDataset  # Import your dataset class

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate sequences if they are too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # Add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"

def main(args):
    # Load model and dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel()  # Instantiate your model
    model.load_state_dict(torch.load(args.model_path))  # Load trained model weights
    model.to(device)

    # Load test data
    test_loader = SpamDataset(args.test_data, batch_size=args.batch_size)  # Assuming SpamDataset accepts the path and batch size

    # Calculate accuracy
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # Plot loss and accuracy if available
    if args.train_losses and args.val_losses:
        epochs_tensor = torch.linspace(0, args.num_epochs, len(args.train_losses))
        examples_seen_tensor = torch.linspace(0, args.examples_seen, len(args.train_losses))
        plot_values(epochs_tensor, examples_seen_tensor, args.train_losses, args.val_losses)

        epochs_tensor = torch.linspace(0, args.num_epochs, len(args.train_accs))
        examples_seen_tensor = torch.linspace(0, args.examples_seen, len(args.train_accs))
        plot_values(epochs_tensor, examples_seen_tensor, args.train_accs, args.val_accs, label="accuracy")

    # Interactive text classification
    while True:
        text = input("Enter a review to classify (or type 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        result = classify_review(text, model, args.tokenizer, device)
        print(f"The review is classified as: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for data loaders")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs for training")
    parser.add_argument("--examples_seen", type=int, required=True, help="Number of examples seen during training")
    parser.add_argument("--train_losses", type=list, default=[], help="List of training losses")
    parser.add_argument("--val_losses", type=list, default=[], help="List of validation losses")
    parser.add_argument("--train_accs", type=list, default=[], help="List of training accuracies")
    parser.add_argument("--val_accs", type=list, default=[], help="List of validation accuracies")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer for encoding text inputs")

    args = parser.parse_args()
    main(args)
