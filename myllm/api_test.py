import torch
import time
import tiktoken
from transformers import GPT2Tokenizer
from config import Config
from api import LLM, GenerationConfig

def pad_sequences(sequences, pad_token=0):
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_token] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)

def benchmark_generation(llm, input_ids, tokenizer, use_cache: bool, max_length: int):
    print(f"\n--- Generating with KV cache = {use_cache}, max_length = {max_length} ---")
    generation_config = GenerationConfig(
        max_length=max_length,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        typical_p=0.9,
        do_sample=True,
        use_kv_cache=use_cache,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
        eos_token_ids=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
        return_tokens=True,
        return_logprobs=False,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False
    )

    start = time.time()
    result = llm.generate(input_ids, generation_config=generation_config)
    elapsed = time.time() - start

    tokens = result["tokens"]
    if tokens.size(0) == 1:
        print("📝 Output:", tokenizer.decode(tokens[0], skip_special_tokens=True))
    else:
        for i, t in enumerate(tokens):
            print(f"📝 Output [{i}]:", tokenizer.decode(t, skip_special_tokens=True))

    print(f"🕒 Generation took {elapsed:.2f} seconds")
    return elapsed

def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_variant = "gpt2-small"

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is defined
    config = Config.from_name(model_variant)
    llm = LLM(config=config, device=device)
    llm.load(model_variant=model_variant, model_family="gpt2")

    # === TEST 1: Prompt lengths ===
    prompts = [
        "Hi",
        "Everything moves you forward.",
        "In a far away land, there lived a mysterious wizard who could control the elements."
    ]
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        print(f"\n=== Testing prompt length: {tokens.shape[1]} tokens ===")
        time_no_cache = benchmark_generation(llm, tokens, tokenizer, use_cache=False, max_length=10)
        time_with_cache = benchmark_generation(llm, tokens, tokenizer, use_cache=True, max_length=10)
        speedup = time_no_cache / time_with_cache if time_with_cache > 0 else float("inf")
        print(f"⚡ KV cache speedup: {speedup:.2f}x")

    # === TEST 2: Generation lengths ===
    prompt = "Everything moves you forward."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for max_len in [10, 50, 100]:
        print(f"\n=== Testing generation length: {max_len} tokens ===")
        time_no_cache = benchmark_generation(llm, input_ids, tokenizer, use_cache=False, max_length=max_len)
        time_with_cache = benchmark_generation(llm, input_ids, tokenizer, use_cache=True, max_length=max_len)
        speedup = time_no_cache / time_with_cache if time_with_cache > 0 else float("inf")
        print(f"⚡ KV cache speedup: {speedup:.2f}x")

    # === TEST 3: Batch generation ===
    batch_prompts = [
        "Everything moves you forward.",
        "Hello world!",
        "Testing KV cache with multiple prompts.",
        "Another example prompt for batch input."
    ]
    batch_tokens = [tokenizer.encode(p) for p in batch_prompts]
    input_ids_batch = pad_sequences(batch_tokens, pad_token=tokenizer.pad_token_id).to(device)

    print(f"\n=== Testing batch size: {input_ids_batch.size(0)} ===")
    time_no_cache = benchmark_generation(llm, input_ids_batch, tokenizer, use_cache=False, max_length=20)
    time_with_cache = benchmark_generation(llm, input_ids_batch, tokenizer, use_cache=True, max_length=20)
    speedup = time_no_cache / time_with_cache if time_with_cache > 0 else float("inf")
    print(f"⚡ KV cache speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()
