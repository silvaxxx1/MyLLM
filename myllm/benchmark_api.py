import os
import time
import torch
import logging
import csv
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
from config import Config
from api import LLM, GenerationConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def benchmark_generation(llm, input_ids, tokenizer, use_cache: bool, max_length: int):
    logging.info(f"Generating with KV cache = {use_cache}, max_length = {max_length}")
    generation_config = GenerationConfig(
        max_length=max_length,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=False,  # Disable sampling for speed and consistency
        use_kv_cache=use_cache,
        eos_token_ids=[tokenizer.eos_token_id],
        return_tokens=True
    )

    start = time.time()
    result = llm.generate(input_ids, generation_config=generation_config)
    elapsed = time.time() - start

    tokens = result["tokens"]
    if tokens.size(0) == 1:
        output_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        logging.info(f"Output: {output_text}")
    else:
        for i, t in enumerate(tokens):
            output_text = tokenizer.decode(t, skip_special_tokens=True)
            logging.info(f"Output [{i}]: {output_text}")

    logging.info(f"Generation took {elapsed:.2f} seconds\n")
    return elapsed

def save_results_to_csv(results, filename="generation_benchmark.csv"):
    keys = results[0].keys() if results else []
    with open(filename, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    logging.info(f"Saved benchmark results to {filename}")

def plot_speedup_bars(results, test_type_filter="generation_length", save_path="kv_cache_speedup.png"):
    filtered = [r for r in results if r["test_type"] == test_type_filter]

    lengths = sorted(set(r["max_length"] for r in filtered))
    speedups = []
    for length in lengths:
        no_cache_time = next((r["time"] for r in filtered if r["max_length"] == length and not r["use_kv_cache"]), None)
        cache_time = next((r["time"] for r in filtered if r["max_length"] == length and r["use_kv_cache"]), None)
        if no_cache_time is None or cache_time is None:
            speedups.append(0)
        else:
            speedups.append(no_cache_time / cache_time if cache_time > 0 else 0)

    # Create bar chart
    plt.figure(figsize=(8,4))
    bars = plt.bar([str(l) for l in lengths], speedups, color='mediumseagreen')
    plt.xlabel("Generation Length (tokens)")
    plt.ylabel("Speedup (No KV Cache / KV Cache)")
    plt.title("KV Cache Speedup by Generation Length")

    # Add value labels on bars
    for bar, speed in zip(bars, speedups):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.1, f"{speed:.2f}×",
                 ha='center', va='top', color='white', fontweight='bold')

    plt.ylim(0, max(speedups) * 1.2)
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f"Saved speedup plot to {save_path}")

def print_speedup_bars(results, test_type_filter="generation_length"):
    print("\nKV Cache Speedup Visualization:\n")
    filtered = [r for r in results if r["test_type"] == test_type_filter]

    lengths = sorted(set(r["max_length"] for r in filtered))
    for length in lengths:
        no_cache_time = next((r["time"] for r in filtered if r["max_length"] == length and not r["use_kv_cache"]), None)
        cache_time = next((r["time"] for r in filtered if r["max_length"] == length and r["use_kv_cache"]), None)
        if no_cache_time is None or cache_time is None:
            continue
        speedup = no_cache_time / cache_time if cache_time > 0 else 0

        max_bar_len = 30
        bar_len = min(int(speedup / 5 * max_bar_len), max_bar_len)
        bar = '▓' * bar_len

        print(f"Gen. Length (tokens) {length:<4}  Speedup (×) {speedup:.2f}  {bar}")

def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_variant = "gpt2-small"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = Config.from_name(model_variant)
    llm = LLM(config=config, device=device)
    llm.load(model_variant=model_variant, model_family="gpt2")

    results = []

    prompts = [
        "Hi",
        "Everything moves you forward."
    ]

    # Test 1: Prompt lengths
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        logging.info(f"Testing prompt length: {tokens.shape[1]} tokens")
        for use_cache in [False, True]:
            elapsed = benchmark_generation(llm, tokens, tokenizer, use_cache=use_cache, max_length=5)
            results.append({
                "test_type": "prompt_length",
                "prompt": prompt,
                "prompt_length": tokens.shape[1],
                "use_kv_cache": use_cache,
                "max_length": 5,
                "time": elapsed
            })

    # Test 2: Generation lengths
    prompt = "Everything moves you forward."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for max_len in [5, 10, 20]:
        logging.info(f"Testing generation length: {max_len} tokens")
        for use_cache in [False, True]:
            elapsed = benchmark_generation(llm, input_ids, tokenizer, use_cache=use_cache, max_length=max_len)
            results.append({
                "test_type": "generation_length",
                "prompt": prompt,
                "prompt_length": input_ids.shape[1],
                "use_kv_cache": use_cache,
                "max_length": max_len,
                "time": elapsed
            })

    # Test 3: Batch generation (optional)
    batch_prompts = [
        "Everything moves you forward.",
        "Hello world!"
    ]
    batch_tokens = [tokenizer.encode(p) for p in batch_prompts]
    input_ids_batch = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in batch_tokens],
        batch_first=True,
        padding_value=tokenizer.eos_token_id
    ).to(device)

    logging.info(f"Testing batch size: {input_ids_batch.size(0)}")
    for use_cache in [False, True]:
        elapsed = benchmark_generation(llm, input_ids_batch, tokenizer, use_cache=use_cache, max_length=10)
        results.append({
            "test_type": "batch_generation",
            "prompt": "batch",
            "prompt_length": input_ids_batch.size(1),
            "use_kv_cache": use_cache,
            "max_length": 10,
            "time": elapsed
        })

    # Save results CSV
    save_results_to_csv(results)

    # Print speedup bars in terminal
    print_speedup_bars(results)

    # Save speedup bar plot as PNG
    plot_speedup_bars(results)

if __name__ == "__main__":
    main()
