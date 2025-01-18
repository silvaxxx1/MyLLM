import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load PPO fine-tuned model and tokenizer
fine_tuned_model_name = "gpt2-imdb-pos-v2"
tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_model_name)
model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_name)

# Ensure pad_token_id is set to avoid warnings
model.config.pad_token_id = model.config.eos_token_id

# Start the conversational loop
print("Conversational loop started. Type 'exit' to end the conversation.")
while True:
    # Get input from the user
    user_input = input("You: ")

    # Exit the loop if user types 'exit'
    if user_input.lower() == "exit":
        print("Exiting conversation...")
        break

    # Tokenize input with attention mask
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)

    # Generate text using the fine-tuned model with sampling parameters
    generated_ids = model.generate(
        inputs['input_ids'], 
        max_length=100, 
        pad_token_id=model.config.pad_token_id, 
        attention_mask=inputs['attention_mask'],
        temperature=0.7,  # Control randomness
        top_p=0.9,        # Nucleus sampling
        top_k=50,         # Top-k sampling
        no_repeat_ngram_size=2,  # Prevent repetition
        do_sample=True    # Use sampling, not greedy decoding
    )

    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Print the model's response
    print("Model: " + generated_text)
