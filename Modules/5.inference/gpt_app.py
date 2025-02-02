import gradio as gr
import torch
import os
import sys
import logging

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from UTILS.generate import TextGenerator  # Import TextGenerator class

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_text_gradio(prompt, length, beams, sampling, temperature):
    """Generates text using the TextGenerator class."""
    logging.info(f"Generating text with prompt: '{prompt[:50]}...' (truncated)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = TextGenerator(model_name="gpt2", device=device)
    
    generated_text = generator.generate(
        prompt=prompt,
        length=length,
        beams=beams if sampling == "greedy" else 1,  # Beams apply only to greedy search
        sampling=sampling,
        temperature=temperature
    )
    
    logging.info("Text generation completed.")
    return generated_text

# Gradio interface components
prompt_input = gr.Textbox(
    label="Enter your Prompt", 
    placeholder="Type your prompt here...", 
    lines=3,  
    max_length=500,  
    info="Provide a short text prompt to guide the text generation."
)

length_input = gr.Slider(
    minimum=10, 
    maximum=500,  
    step=1, 
    label="Length of Generated Text", 
    value=100,  
    info="Control how long the generated text should be (in terms of tokens/words)."
)

beams_input = gr.Slider(
    minimum=1, 
    maximum=10,  
    step=1, 
    label="Number of Beams (Beam Search)", 
    value=3, 
    info="Set the number of beams for beam search, affecting the quality of the output."
)

sampling_input = gr.Radio(
    ["greedy", "top_k", "nucleus"],  
    label="Sampling Method", 
    value="greedy",  
    info="Choose the sampling method. 'Greedy' selects the most probable token, 'Top-k' limits the possible next tokens, 'Nucleus' samples from the top 'p' probability."
)

temperature_input = gr.Slider(
    minimum=0.0, 
    maximum=1.0, 
    step=0.1, 
    label="Temperature (Randomness Control)", 
    value=0.7,  
    info="Adjust the temperature: higher values make the output more random, lower values make it more deterministic."
)

output_text = gr.Textbox(
    label="Generated Text", 
    placeholder="Generated content will appear here...", 
    lines=10,  
    interactive=False,  
    info="This is where the generated text will be displayed based on your inputs."
)

# Create the Gradio interface
app = gr.Interface(
    fn=generate_text_gradio,  
    inputs=[prompt_input, length_input, beams_input, sampling_input, temperature_input],  
    outputs=output_text,  
    live=True,  
    title="Meta_Bot Demo",  
    description="This tool allows you to generate creative and customizable text based on your prompt. "
                "Adjust parameters like text length, sampling strategy, and randomness (temperature) to control the output. "
                "Perfect for exploring different types of text generation based on various configurations.",
    theme="default"  # Changed from "compact" to "default"
)

if __name__ == "__main__":
    logging.info("Launching Meta_Bot Demo...")
    app.launch(share=False)  # Enables a public link
