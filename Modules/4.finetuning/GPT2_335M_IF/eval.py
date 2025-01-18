import json
import urllib.request
from tqdm import tqdm
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor
import tiktoken
from utils import generate, generate_text_simple, text_to_tokens_ids, token_ids_to_text
from data import format_input
import torch 

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_if_running(process_name):
    """Check if a specific process is running."""
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            return True
    return False

# Check if Ollama is running
ollama_running = check_if_running("ollama")
if not ollama_running:
    raise RuntimeError("Ollama not running. Launch Ollama before proceeding.")

logging.info("Ollama is running.")

def query_model(prompt, model="llama3.2", url="http://localhost:11434/api/chat"):
    """Query the Ollama model with the given prompt."""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }
    
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    
    response_data = ""
    try:
        with urllib.request.urlopen(request) as response:
            while True:
                line = response.readline().decode("utf-8")
                if not line:
                    break
                response_json = json.loads(line)
                response_data += response_json["message"]["content"]
    except Exception as e:
        logging.error(f"Error querying model: {e}")
        return None

    return response_data.strip()

def generate_model_scores(json_data, json_key, model="llama3.2"):
    """Generate scores for model responses based on provided data."""
    scores = []
    
    # Using ThreadPoolExecutor to parallelize the scoring process
    with ThreadPoolExecutor() as executor:
        futures = []
        for entry in tqdm(json_data, desc="Scoring entries"):
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}` "
                f"on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            futures.append(executor.submit(query_model, prompt, model))
        
        for future in tqdm(futures, desc="Collecting scores"):
            score = future.result()
            if score is not None:
                try:
                    scores.append(int(score))
                except ValueError:
                    logging.warning(f"Could not convert score: {score}")
                    scores.append(None)  # Append None for invalid scores

    return scores

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_model_responses(test_data, model, tokenizer, context_length):
    """Generate responses for the model based on test data."""
    for i, entry in tqdm(enumerate(test_data), total=len(test_data), desc="Generating responses"):
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_tokens_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=context_length,
            eos_id=50256
        )

        gen_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (gen_text[len(input_text):].replace("### Response:", "").strip())
        test_data[i]["model_response"] = response_text

    # Save the responses to a JSON file
    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)
    logging.info("Generated responses saved to instruction-data-with-response.json")

def main(input_file, json_key, model):
    """Main function to load data, generate scores, and save results."""
    # Load JSON data
    with open(input_file, 'r') as f:
        json_data = json.load(f)

    # Generate scores
    scores = generate_model_scores(json_data, json_key, model)

    # Save scores to a JSON file
    output_file = input_file.replace('.json', '_scores.json')
    with open(output_file, 'w') as f:
        json.dump(scores, f)

    logging.info(f"Scores saved to {output_file}")

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
    BASE_CONFIG.update(model_configs[model])

    # Generate model responses for test data
    tokenizer = tiktoken.get_encoding("gpt2")
    generate_model_responses(json_data, model, tokenizer, BASE_CONFIG["context_length"])

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model responses using Ollama.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file containing data for evaluation.")
    parser.add_argument("--json_key", type=str, default="model_response", help="Key for the model response in the JSON data.")
    parser.add_argument("--model", type=str, default="llama3.2", help="Model name to query.")
    
    args = parser.parse_args()
    main(args.input_file, args.json_key, args.model)
