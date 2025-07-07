import sys
import os
import torch
import pytest
import tiktoken  # ✅ ADD THIS

# Add the parent directory to the sys.path so we can import api, config, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from api import LLM, GenerationConfig


@pytest.mark.parametrize("use_kv_cache", [False, True])
def test_llm_generate(use_kv_cache):
    # Select model and device
    model_variant = "gpt2-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config and initialize model
    config = Config.from_name(model_variant)
    llm = LLM(config=config, device=device)
    llm.load(model_variant=model_variant, model_family="gpt2")

    # ✅ Use tiktoken tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Hello"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

    eos_token = tokenizer.eot_token  # ✅ Use tiktoken's EOS token ID

    # Generation config
    generation_config = GenerationConfig(
        max_length=20,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        do_sample=True,
        use_kv_cache=use_kv_cache,
        repetition_penalty=1.0,
        eos_token_id=eos_token,
        pad_token_id=eos_token
    )

    # Generate output
    output = llm.generate(input_ids=input_ids, generation_config=generation_config)

    # Assertions
    assert output.shape[0] == input_ids.shape[0], "Batch size mismatch"
    assert output.shape[1] <= generation_config.max_length, "Exceeded max_length"
    assert output.dtype == torch.long, "Output should be long dtype"

    decoded = tokenizer.decode(output[0].tolist())
    print("📝 Output:", decoded)
