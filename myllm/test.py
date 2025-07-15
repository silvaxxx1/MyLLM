import torch
from Configs.ModelConfig import ModelConfig
from Configs.GenConfig import GenerationConfig
from api import LLM
# Step 1: Choose your device
device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import GPT2Tokenizer
# Step 2: Load the model config
model_config = ModelConfig.from_name("gpt2-small")  # or any variant

# Step 3: Create the LLM instance
llm = LLM(config=model_config, device=device)

# Step 4: Load weights (downloaded automatically if needed)
llm.load(model_variant="gpt2-small")  # Will download .safetensors if missing

# Step 5: Define generation config
gen_cfg = GenerationConfig(
    max_length=50,
    temperature=1.0,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    use_optimized_sampler=True,
    use_kv_cache=True,
    return_tokens=True,
    return_logprobs=True,
    eos_token_ids=[50256],  # adjust for your tokenizer
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Step 6: Generate text
prompt = "In a distant galaxy"
output = llm.generate_text(prompt, tokenizer, gen_cfg)

print("Generated text:\n", output["text"])
