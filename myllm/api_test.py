import torch
from config import Config
from api import LLM
from transformers import GPT2Tokenizer

def main():
    model_variant = "gpt2-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config.from_name(model_variant)
    llm = LLM(config=config, device=device)

    # Load weights (efficient param removed)
    llm.load(model_variant=model_variant, model_family="gpt2")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_text = "Everything move you forwards"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    generated = llm.generate(input_ids, generation_config={
        "max_length": 20,
        "temperature": 1.0
    })

    print("🔢 Generated token IDs:", generated)
    print("📝 Generated text:", tokenizer.decode(generated[0].tolist(), skip_special_tokens=True))

if __name__ == "__main__":
    main()
