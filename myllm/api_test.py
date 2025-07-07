from transformers import GPT2Tokenizer
from config import Config
from api import LLM, GenerationConfig
import torch
import time

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = "In a far away land, there lived a mysterious wizard who"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    # Initialize model
    config = Config.from_name("gpt2-medium")
    llm = LLM(config=config, device="cuda")
    llm.load(model_variant="gpt2-medium", model_family="gpt2")

    # Generation config WITHOUT KV cache
    gen_config_no_cache = GenerationConfig(
        max_length=50,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        use_kv_cache=False,
        eos_token_id=tokenizer.eos_token_id
    )

    # Generation config WITH KV cache
    gen_config_with_cache = GenerationConfig(
        max_length=50,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        use_kv_cache=True,
        eos_token_id=tokenizer.eos_token_id
    )

    # 🚫 Generate WITHOUT KV cache
    print("🚫 Generating without KV cache...")
    start = time.time()
    output_ids1 = llm.generate(input_ids, generation_config=gen_config_no_cache)
    duration_no_cache = round(time.time() - start, 2)
    print(f"🕒 Time (no cache): {duration_no_cache} sec")
    print("📝 Output (no cache):\n" + tokenizer.decode(output_ids1[0], skip_special_tokens=True).strip())

    # ⚡ Generate WITH KV cache
    print("\n⚡ Generating with KV cache...")
    start = time.time()
    output_ids2 = llm.generate(input_ids, generation_config=gen_config_with_cache)
    duration_cache = round(time.time() - start, 2)
    print(f"🕒 Time (with cache): {duration_cache} sec")
    print("📝 Output (with cache):\n" + tokenizer.decode(output_ids2[0], skip_special_tokens=True).strip())

if __name__ == "__main__":
    main()
