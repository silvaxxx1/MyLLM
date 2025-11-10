# ===============================
# Core dependencies
# ===============================
import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

# ===============================
# Local modules
# ===============================
from .model import GPT
from .Configs import ModelConfig, GenerationConfig
from .utils import OptimizedSampler

# Unified loader and weight mappers
from .utils.loader import ModelLoader
from .utils import WEIGHT_MAPPERS

# ===============================
# Main LLM Class
# ===============================
class LLM(nn.Module):
    """
    Wrapper around GPT models with memory-optimized loading and generation.
    """
    def __init__(self,
                 config: ModelConfig = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 low_cpu_mem_usage: bool = True,
                 device: str = "cpu"):
        super().__init__()
        self.device = device
        self.config = config
        self.model: Optional[nn.Module] = None
        self.loader = ModelLoader(cache_dir="./models")
        self.torch_dtype = torch_dtype or torch.float32
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.sampler = OptimizedSampler()
        self.use_amp = torch.cuda.is_available() and device != "cpu"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        if config is not None:
            self.init_from_config(config)

    def init_from_config(self, config: ModelConfig, train_mode: bool = True):
        """Initialize GPT model structure without loading full weights to GPU"""
        self.config = config
        self.model = GPT(config)
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        if self.low_cpu_mem_usage:
            print("⚡ Low-memory mode: model initialized on CPU")
        else:
            self.model.to(self.device, dtype=self.torch_dtype)

    def load(self, model_variant: str, model_family: Optional[str] = None):
        """Load model weights using memory-optimized loader"""
        # Use the loader to automatically detect family if not provided
        self.model, self.config = self.loader.load(
            model_variant=model_variant,
            device=self.device,
            model_family=model_family,
            custom_config=self.config,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=self.low_cpu_mem_usage
        )

        self.model.eval()
        torch.cuda.empty_cache()
        print(f"🎯 Model ready on {self.device} for inference!")

    def list_models(self):
        """List available models in cache"""
        return self.loader.list_available_models()

    def save(self, save_path: str):
        """Save model weights to disk"""
        if self.model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved to {save_path}")

    # -------------------------
    # Generation utilities
    # -------------------------
    def _setup_generation(self, input_ids: torch.Tensor, generation_config: GenerationConfig):
        B, T = input_ids.shape
        if generation_config.use_kv_cache:
            self.model.reset_cache()
            max_seq_len = T + generation_config.max_length
            self.model.initialize_kv_cache(batch_size=B, max_seq_len=max_seq_len)

            with torch.no_grad():
                if generation_config.use_mixed_precision and self.use_amp:
                    with torch.cuda.amp.autocast():
                        _ = self.model(input_ids, use_cache=False, pos_offset=0)
                else:
                    _ = self.model(input_ids, use_cache=False, pos_offset=0)

        return B, T

    def _forward_step(self, input_token: torch.Tensor, use_cache: bool, pos_offset: int,
                      use_mixed_precision: bool) -> torch.Tensor:
        if use_mixed_precision and self.use_amp:
            with torch.cuda.amp.autocast():
                return self.model(input_token, use_cache=use_cache, pos_offset=pos_offset)
        return self.model(input_token, use_cache=use_cache, pos_offset=pos_offset)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, generation_config: GenerationConfig) -> Dict[str, Any]:
        self.model.eval()
        B, T = self._setup_generation(input_ids, generation_config)
        generated = input_ids.clone()
        logprobs = []
        max_logprob_history = generation_config.max_logprob_history
        eos_tensor = torch.tensor(generation_config.eos_token_ids, device=input_ids.device) \
            if generation_config.eos_token_ids else None

        for step in range(generation_config.max_length):
            input_token = generated[:, -1:] if generation_config.use_kv_cache else generated
            pos_offset = self.model.transformer["block_0"].attn.kv_cache.size if generation_config.use_kv_cache else 0

            logits = self._forward_step(
                input_token, generation_config.use_kv_cache,
                pos_offset, generation_config.use_mixed_precision
            )

            next_token_logits = logits[:, -1, :] / generation_config.temperature

            # Apply optimized sampler
            if generation_config.use_optimized_sampler:
                if generation_config.apply_repetition_penalty and generation_config.repetition_penalty != 1.0:
                    next_token_logits = self.sampler.apply_repetition_penalty_vectorized(
                        next_token_logits, generated, generation_config.repetition_penalty
                    )

                top_k = generation_config.top_k if generation_config.apply_top_k_sampling else None
                top_p = generation_config.top_p if generation_config.apply_top_p_sampling else None

                next_token_logits = self.sampler.combined_top_k_top_p_sampling(
                    next_token_logits, top_k, top_p
                )

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1) if generation_config.do_sample else \
                         torch.argmax(probs, dim=-1, keepdim=True)

            if generation_config.return_logprobs:
                logprob = torch.log(probs.gather(1, next_token))
                logprobs.append(logprob)
                if len(logprobs) > max_logprob_history:
                    logprobs.pop(0)

            generated = torch.cat([generated, next_token], dim=1)

            if generation_config.early_stopping and eos_tensor is not None:
                if self.sampler.check_eos_vectorized(next_token, generation_config.eos_token_ids):
                    break

        output = {"tokens": generated}
        if generation_config.return_logprobs and logprobs:
            output["logprobs"] = torch.cat(logprobs, dim=1)
        return output

    def generate_text(self, prompt: str, tokenizer, generation_config: GenerationConfig) -> Dict[str, Any]:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.generate(input_ids, generation_config)
        text = tokenizer.decode(output["tokens"][0], skip_special_tokens=True)
        return {
            "text": text,
            "tokens": output["tokens"][0].tolist() if generation_config.return_tokens else None,
            "logprobs": output.get("logprobs", None)
        }

    def generate_batch(self, prompts: List[str], tokenizer, generation_config: GenerationConfig) -> List[Dict[str, Any]]:
        if len(prompts) == 0:
            return []

        encoded = [tokenizer.encode(p, return_tensors="pt") for p in prompts]
        max_len = max(e.shape[1] for e in encoded)

        padded = []
        for e in encoded:
            pad_len = max_len - e.shape[1]
            pad_token_id = generation_config.pad_token_id or tokenizer.pad_token_id or 0
            if pad_len > 0:
                padding = torch.full((1, pad_len), pad_token_id, dtype=e.dtype)
                e = torch.cat([e, padding], dim=1)
            padded.append(e)

        batch_input = torch.cat(padded, dim=0).to(self.device)
        batch_output = self.generate(batch_input, generation_config)
        results = []

        for i in range(len(prompts)):
            tokens = batch_output["tokens"][i]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            result = {
                "text": text,
                "tokens": tokens.tolist() if generation_config.return_tokens else None,
                "logprobs": batch_output.get("logprobs", [None] * len(prompts))[i] if generation_config.return_logprobs else None
            }
            results.append(result)

        return results

###############################################
if __name__ == "__main__":
    from myllm.Tokenizers.factory import get_tokenizer

    # Pick the correct tokenizer
    tokenizer_name = "gpt2"  # instead of "gpt2"
    tokenizer = get_tokenizer(tokenizer_name)

    # Pick a model variant that exists
    model_config = ModelConfig.from_name("gpt2-xl")  # must match model_registry.py

    device = "cuda" if torch.cuda.is_available() else "cpu"

    llm = LLM(
        config=model_config,
        device=device,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    llm.load("gpt2-xl")  # must match model_registry.py
    torch.cuda.empty_cache()

    prompt = "Once upon a time"
    generation_config = GenerationConfig(
        max_length=50,
        temperature=0.8,
        use_kv_cache=True,
        use_mixed_precision=True
    )

    result = llm.generate_text(prompt, tokenizer, generation_config)
    print(result["text"]) 
