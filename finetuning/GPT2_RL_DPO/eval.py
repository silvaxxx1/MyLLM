import argparse
import pickle
import torch
from utils.finetune_utils import plot_losses
from utils.finetune_utils import generate, text_to_token_ids, token_ids_to_text
from data import format_input
import tiktoken



# Define function to save tracking data
def save_tracking_data(tracking, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(tracking, f)

# Define function to load tracking data
def load_tracking_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Define function to plot losses
def plot_tracking_losses(tracking, num_epochs):
    epochs_tensor = torch.linspace(0, num_epochs, len(tracking["train_losses"]))
    
    plot_losses(
        epochs_seen=epochs_tensor,
        tokens_seen=tracking["tokens_seen"],
        train_losses=tracking["train_losses"],
        val_losses=tracking["val_losses"],
        label="loss"
    )
    
    train_reward_margins = [i-j for i,j in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
    val_reward_margins = [i-j for i,j in zip(tracking["val_chosen_rewards"], tracking["val_rejected_rewards"])]
    
    plot_losses(
        epochs_seen=epochs_tensor,
        tokens_seen=tracking["tokens_seen"],
        train_losses=train_reward_margins,
        val_losses=val_reward_margins,
        label="reward margins"
    )

# Define function to generate model responses and print results
def generate_and_print_responses(data, ref_model, policy_model, tokenizer, device, base_config):
    for entry in data[:3]:
        input_text = format_input(entry)

        token_ids = generate(
            model=ref_model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=base_config["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        reference_response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )

        token_ids = generate(
            model=policy_model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=base_config["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        policy_response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nReference model response:\n>> {reference_response_text.strip()}")
        print(f"\nPolicy model response:\n>> {policy_response_text.strip()}")
        print("\n-------------------------------------\n")

# Main function to automate the process
def main(args):
    # Initialize tokenizer
    tokenizer = tiktoken.GPT2Tokenizer.from_pretrained("gpt2")
    
    # Dummy tracking data (can be replaced by your actual data)
    tracking = {
        "train_losses": [0.5, 0.4, 0.3],
        "val_losses": [0.6, 0.5, 0.4],
        "tokens_seen": [1000, 2000, 3000],
        "train_chosen_rewards": [1, 1, 1],
        "train_rejected_rewards": [0.5, 0.5, 0.5],
        "val_chosen_rewards": [1, 1, 1],
        "val_rejected_rewards": [0.5, 0.5, 0.5],
    }

    # Save tracking data
    save_tracking_data(tracking, args.tracking_file)
    
    # Load tracking data
    tracking = load_tracking_data(args.tracking_file)
    
    # Plot losses
    plot_tracking_losses(tracking, num_epochs=2)
    
    # Set the device and models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ref_model = None  # Initialize your reference model here
    policy_model = None  # Initialize your policy model here
    base_config = {"context_length": 512}  # Set your base config

    # Generate and print responses (use val_data and test_data as needed)
    generate_and_print_responses(args.val_data, ref_model, policy_model, tokenizer, device, base_config)
    generate_and_print_responses(args.test_data, ref_model, policy_model, tokenizer, device, base_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automate model training and evaluation")
    
    parser.add_argument('--tracking_file', type=str, required=True, help="Path to the tracking file")
    parser.add_argument('--val_data', type=str, required=True, help="Path to validation data")
    parser.add_argument('--test_data', type=str, required=True, help="Path to test data")
    
    args = parser.parse_args()
    
    main(args)
