from myllm import LLM
from myllm.Configs import ModelConfig, GenerationConfig
from myllm.Tokenizers.factory import get_tokenizer
from .test import run_enhanced_diagnostics  # Updated import
import torch

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = get_tokenizer("gpt2")
model_config = ModelConfig.from_name("gpt2-small")

llm = LLM(
    config=model_config,
    device=device,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
llm.load("gpt2-small")

generation_config = GenerationConfig(
    max_length=50,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    use_kv_cache=True,
    use_mixed_precision=True
)

# Run enhanced diagnostics
results = run_enhanced_diagnostics(llm, tokenizer, generation_config, device)