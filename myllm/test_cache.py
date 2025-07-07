import torch
import time
import tiktoken
from config import Config
from api import LLM

def pad_sequences(sequences, pad_token=0):
    # Pad list of token lists to equal length tensor
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_token] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)

def benchmark_generation(llm, input_ids, encoder, use_cache: bool, max_length: int):
    print(f"\n--- Generating with KV cache = {use_cache}, max_length = {max_length} ---")
    start = time.time()
    generated = llm.generate(
        input_ids,
        generation_config={"max_length": max_length, "temperature": 1.0},
        use_cache=use_cache
    )
    elapsed = time.time() - start
    print("Generated token IDs:", generated)
    # For batch inputs, decode each separately
    if generated.size(0) == 1:
        print("Generated text:", encoder.decode(generated[0].tolist()))
    else:
        for i, gen_tokens in enumerate(generated):
            print(f"Generated text [{i}]:", encoder.decode(gen_tokens.tolist()))
    print(f"Generation took {elapsed:.4f} seconds")
    return elapsed

def main():
    
    torch.manual_seed(42)
    model_variant = "gpt2-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config.from_name(model_variant)
    llm = LLM(config=config, device=device)
    llm.load(model_variant=model_variant, model_family="gpt2")

    encoder = tiktoken.get_encoding("gpt2")

    # Test 1: Different prompt lengths
    prompts = [
        "Hi",  # short prompt
        "Everything move you forwards ",  # medium prompt
        "In a far away land, there lived a mysterious wizard who could control the elements. "  # long prompt
    ]

    for prompt in prompts:
        tokens = encoder.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)
        print(f"\n=== Testing prompt length: {len(tokens)} tokens ===")
        time_no_cache = benchmark_generation(llm, input_ids, encoder, use_cache=False, max_length=10)
        time_with_cache = benchmark_generation(llm, input_ids, encoder, use_cache=True, max_length=10)
        speedup = time_no_cache / time_with_cache if time_with_cache > 0 else float('inf')
        print(f"KV cache speedup: {speedup:.2f}x")

    # Test 2: Longer generation lengths
    prompt = "Everything move you forwards "
    tokens = encoder.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)
    for max_len in [10, 50, 100]:
        print(f"\n=== Testing generation length: {max_len} tokens ===")
        time_no_cache = benchmark_generation(llm, input_ids, encoder, use_cache=False, max_length=max_len)
        time_with_cache = benchmark_generation(llm, input_ids, encoder, use_cache=True, max_length=max_len)
        speedup = time_no_cache / time_with_cache if time_with_cache > 0 else float('inf')
        print(f"KV cache speedup: {speedup:.2f}x")

    # Test 3: Batch generation
    batch_prompts = [
        "Everything move you forwards ",
        "Hello world!",
        "Testing KV cache with multiple prompts.",
        "Another example prompt for batch input."
    ]
    batch_tokens = [encoder.encode(p) for p in batch_prompts]
    input_ids_batch = pad_sequences(batch_tokens, pad_token=encoder.eot_token).to(device)

    print(f"\n=== Testing batch size: {input_ids_batch.size(0)} ===")
    time_no_cache = benchmark_generation(llm, input_ids_batch, encoder, use_cache=False, max_length=20)
    time_with_cache = benchmark_generation(llm, input_ids_batch, encoder, use_cache=True, max_length=20)
    speedup = time_no_cache / time_with_cache if time_with_cache > 0 else float('inf')
    print(f"KV cache speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()
