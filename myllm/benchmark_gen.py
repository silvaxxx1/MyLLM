import time
import torch
from transformers import GPT2Tokenizer
from config import Config
from gen import LLM, GenerationConfig

def benchmark_generation(llm, input_ids, tokenizer, use_cache: bool, max_length: int):
    print(f"\nBenchmarking: KV cache = {use_cache}, max_length = {max_length}")
    generation_config = GenerationConfig(
        max_length=max_length,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=False,
        use_kv_cache=use_cache,
        eos_token_ids=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
        return_tokens=True,
    )

    start = time.time()
    output = llm.generate(input_ids, generation_config)
    elapsed = time.time() - start

    tokens = output["tokens"]
    if tokens.size(0) == 1:
        print("Output:", tokenizer.decode(tokens[0], skip_special_tokens=True))
    else:
        for i, t in enumerate(tokens):
            print(f"Output [{i}]:", tokenizer.decode(t, skip_special_tokens=True))

    print(f"Time taken: {elapsed:.3f} seconds")
    return elapsed

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_variant = "gpt2-small"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = Config.from_name(model_variant)

    llm = LLM(config=config, device=device)
    llm.load(model_variant=model_variant, model_family="gpt2")

    # Test prompt lengths
    prompts = [
        "Hi",
        "Everything moves you forward.",
        "Once upon a time, in a distant galaxy, far away from Earth,"
    ]

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        print(f"\nPrompt length: {input_ids.shape[1]} tokens")
        no_cache_time = benchmark_generation(llm, input_ids, tokenizer, use_cache=False, max_length=10)
        cache_time = benchmark_generation(llm, input_ids, tokenizer, use_cache=True, max_length=10)
        speedup = no_cache_time / cache_time if cache_time > 0 else float('inf')
        print(f"KV cache speedup: {speedup:.2f}x")

    # Test generation lengths
    prompt = "Everything moves you forward."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for max_len in [10, 50, 100]:
        print(f"\nGeneration length: {max_len} tokens")
        no_cache_time = benchmark_generation(llm, input_ids, tokenizer, use_cache=False, max_length=max_len)
        cache_time = benchmark_generation(llm, input_ids, tokenizer, use_cache=True, max_length=max_len)
        speedup = no_cache_time / cache_time if cache_time > 0 else float('inf')
        print(f"KV cache speedup: {speedup:.2f}x")

    # Test batch generation
    batch_prompts = [
        "Everything moves you forward.",
        "Hello world!",
        "Testing KV cache with multiple prompts.",
        "Another example prompt for batch input."
    ]

    batch_tokens = [tokenizer.encode(p) for p in batch_prompts]
    max_len = max(len(t) for t in batch_tokens)
    padded_tokens = [t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in batch_tokens]
    input_ids_batch = torch.tensor(padded_tokens, dtype=torch.long).to(device)

    print(f"\nBatch size: {input_ids_batch.size(0)}")
    no_cache_time = benchmark_generation(llm, input_ids_batch, tokenizer, use_cache=False, max_length=20)
    cache_time = benchmark_generation(llm, input_ids_batch, tokenizer, use_cache=True, max_length=20)
    speedup = no_cache_time / cache_time if cache_time > 0 else float('inf')
    print(f"KV cache speedup (batch): {speedup:.2f}x")

if __name__ == "__main__":
    main()
