import torch
from Tokenizers.factory import get_tokenizer, list_available_models
from Tokenizers.wrapper import TokenizerWrapper
from api import LLM
from Configs import ModelConfig, GenerationConfig

def test_basic_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_variant = "gpt2-small"

    print("Available tokenizers:", list_available_models())

    # Wrap the tokenizer
    tokenizer_raw = get_tokenizer("gpt2")
    tokenizer = TokenizerWrapper(tokenizer_raw)

    # Ensure pad token
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else 0

    config = ModelConfig.from_name(model_variant)
    llm = LLM(config=config, device=device)
    llm.load(model_variant=model_variant, model_family="gpt2")

    prompts = [
        "Hello world!",
        "In the beginning,",
        "Once upon a time, in a land far away,"
    ]

    gen_config = GenerationConfig(
        max_length=20,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=False,
        use_kv_cache=True,
        repetition_penalty=1.2,
        early_stopping=True,
        return_tokens=True,
        return_logprobs=False,
    )

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)           # returns list
        input_tensor = torch.tensor([input_ids], device=device)
        output = llm.generate(input_tensor, gen_config)
        tokens = output["tokens"][0].tolist()          # tensor → list
        text = tokenizer.decode(tokens)
        print(f"Prompt: {prompt}")
        print(f"Output: {text}")
        assert len(tokens) > len(input_ids), "No tokens generated"
        assert isinstance(text, str), "Output text must be a string"

def test_batch_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_variant = "gpt2-small"

    tokenizer_raw = get_tokenizer("gpt2")
    tokenizer = TokenizerWrapper(tokenizer_raw)

    # Ensure pad token
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else 0

    config = ModelConfig.from_name(model_variant)
    llm = LLM(config=config, device=device)
    llm.load(model_variant=model_variant, model_family="gpt2")

    batch_prompts = [
        "Hello world!",
        "In the beginning,",
        "Once upon a time, in a land far away,"
    ]

    gen_config = GenerationConfig(
        max_length=20,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=False,
        use_kv_cache=True,
        repetition_penalty=1.2,
        early_stopping=True,
        return_tokens=True,
        return_logprobs=False,
    )

    results = llm.generate_batch(batch_prompts, tokenizer, gen_config)

    assert len(results) == len(batch_prompts), "Batch results count mismatch"

    for prompt, res in zip(batch_prompts, results):
        tokens_tensor = res["tokens"]
        tokens = tokens_tensor.tolist() if torch.is_tensor(tokens_tensor) else tokens_tensor
        text = tokenizer.decode(tokens)
        print(f"Prompt: {prompt}")
        print(f"Output: {text}")
        assert isinstance(text, str), "Batch output text must be a string"
        assert len(tokens) > 0, "Batch output tokens missing"

if __name__ == "__main__":
    print("Running basic generation tests...")
    test_basic_generation()
    print("Running batch generation tests...")
    test_batch_generation()
    print("All tests passed!")
