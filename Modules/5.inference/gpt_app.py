import torch
import tiktoken
import gradio as gr

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Load the generate_text function from your code
from training.train_utils import generate_text
from configs.gpt_config import GPT_CONFIG_124M, GPT_CONFIG_355M, model_names

# Function to load the model
def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

# Encode the input prompt
def encode_prompt(prompt, tokenizer):
    return tokenizer.encode(prompt)

# Decode generated tokens
def decode_tokens(token_ids, tokenizer):
    return tokenizer.decode(token_ids)

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Map model names to configuration and model paths
model_config_map = {
    "gpt2-small (124M)": (GPT_CONFIG_124M, r"C:\Users\user\Documents\SILVA AI ROADMAP\MyLLM\inference\gpt_infrence\checkpoints\models\gpt2-small (124M)_model.pt"),
    "gpt2-medium (355M)": (GPT_CONFIG_355M, r"C:\Users\user\Documents\SILVA AI ROADMAP\MyLLM\inference\gpt_infrence\checkpoints\models\gpt2-medium (355M)_model.pt"),
}

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a function for the Gradio UI that uses your generate_text function
def generate_text_ui(prompt, model_name, max_length, temperature, top_k):
    config, model_save_path = model_config_map[model_name]
    model = load_model(model_save_path, device)

    input_ids = torch.tensor(encode_prompt(prompt, tokenizer)).unsqueeze(0).to(device)

    # Generate text using your existing function
    output_ids = generate_text(
        model=model,
        idx=input_ids,
        max_new_tokens=max_length,
        context_size=config['context_length'],
        temperature=temperature,
        top_k=top_k,
        eos_id=50256
    )

    # Decode the generated tokens
    generated_text = decode_tokens(output_ids.squeeze().tolist(), tokenizer)
    
    return generated_text

# Create Gradio Interface
# Create Gradio Interface
def interface():
    gr.Interface(
        fn=generate_text_ui, 
        inputs=[
            gr.Textbox(lines=2, label="Enter your prompt"),
            gr.Dropdown(list(model_config_map.keys()), label="Select Model"),
            gr.Slider(10, 200, step=1, value=100, label="Max New Tokens"),  # Changed default to value
            gr.Slider(0.0, 1.5, step=0.1, value=0.7, label="Temperature"),  # Changed default to value
            gr.Slider(1, 100, step=1, value=50, label="Top K")  # Changed default to value
        ], 
        outputs=gr.Textbox(label="Generated Text"),
        title="SILVA GPT App",
        description="Generate text using different models and parameters",
        theme="default"
    ).launch(share=True)



# Run the interface
if __name__ == "__main__":
    interface()

