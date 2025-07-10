import torch
from transformers import GPT2Tokenizer
from config import Config
from api import LLM, GenerationConfig

def test_basic_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_variant = "gpt2-small"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = Config.from_name(model_variant)

    llm = LLM(config=config, device=device)
    llm.load(model_variant=model_variant, model_family="gpt2")

    prompts = [
        "Hello world!",
        "In the beginning,",
        "Once upon a time, in a land far away,"
    ]

    generation_config = GenerationConfig(
        max_length=20,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=False,
        use_kv_cache=True,
        repetition_penalty=1.2,
        early_stopping=True,
        eos_token_ids=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
        return_tokens=True,
        return_logprobs=False,
    )

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = llm.generate(input_ids, generation_config)
        tokens = output["tokens"][0]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Output: {text}")
        assert len(tokens) > input_ids.shape[1], "No tokens generated"
        assert isinstance(text, str), "Output text must be a string"

def test_batch_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_variant = "gpt2-small"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = Config.from_name(model_variant)

    llm = LLM(config=config, device=device)
    llm.load(model_variant=model_variant, model_family="gpt2")

    batch_prompts = [
        "Hello world!",
        "In the beginning,",
        "Once upon a time, in a land far away,"
    ]

    generation_config = GenerationConfig(
        max_length=20,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=False,
        use_kv_cache=True,
        repetition_penalty=1.2,
        early_stopping=True,
        eos_token_ids=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
        return_tokens=True,
        return_logprobs=False,
    )

    results = llm.generate_batch(batch_prompts, tokenizer, generation_config)
    assert len(results) == len(batch_prompts), "Batch results count mismatch"

    for prompt, res in zip(batch_prompts, results):
        print(f"Prompt: {prompt}")
        print(f"Output: {res['text']}")
        assert isinstance(res["text"], str), "Batch output text must be a string"
        assert len(res["tokens"]) > 0, "Batch output tokens missing"

if __name__ == "__main__":
    print("Running basic generation tests...")
    test_basic_generation()
    print("Running batch generation tests...")
    test_batch_generation()
    print("All tests passed!")
