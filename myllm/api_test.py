import torch
from config import Config
from api import LLM
from transformers import GPT2Tokenizer

def main():
    config = Config.from_name("gpt2-medium")
    device = "cpu"

    llm = LLM(config=config, device=device)

    # ✅ Load with efficient sequential loading
    llm.load(model_variant="gpt2-medium", model_family="gpt2", efficient=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    input_text = "Hello"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    generated = llm.generate(input_ids, generation_config={"max_length": 20, "temperature": 0.8})

    print("Generated token IDs:", generated)
    print("Generated text:", tokenizer.decode(generated[0].tolist()))

if __name__ == "__main__":
    main()
