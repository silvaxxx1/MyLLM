
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