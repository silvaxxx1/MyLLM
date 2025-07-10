# Core dependencies
import os
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from Configs.GenConfig import GenerationConfig 
# Local modules
from model import GPT
from Configs.ModelConfig import ModelConfig
from utils.download_weight import (
    download_safetensors,
    load_safetensors,
    load_gpt2_weights_meta,
    get_gpt2_safetensors_url,
    Spinner
)


# ===============================
# Vectorized Sampler Utilities
# ===============================

class OptimizedSampler:
    """
    Provides efficient sampling utilities for decoding strategies.
    """

    @staticmethod
    def apply_repetition_penalty_vectorized(logits: torch.Tensor, 
                                            generated_tokens: torch.Tensor, 
                                            penalty: float) -> torch.Tensor:
        """
        Apply repetition penalty to logits using vectorized operations.
        """
        if penalty == 1.0:
            return logits

        batch_size, vocab_size = logits.shape
        penalty_mask = torch.ones_like(logits)

        # Penalize previously generated tokens per batch
        for i in range(batch_size):
            unique_tokens = generated_tokens[i].unique()
            penalty_mask[i, unique_tokens] = 1.0 / penalty

        return logits * penalty_mask

    @staticmethod
    def combined_top_k_top_p_sampling(logits: torch.Tensor, 
                                      top_k: Optional[int] = None, 
                                      top_p: Optional[float] = None) -> torch.Tensor:
        """
        Apply both top-k and top-p sampling for probabilistic decoding.
        """
        if top_k is None and top_p is None:
            return logits

        batch_size, vocab_size = logits.shape

        if top_k is not None:
            k = min(top_k, vocab_size)
            values, indices = torch.topk(logits, k, dim=-1)

            if top_p is not None:
                probs = torch.softmax(values, dim=-1)
                cumsum = torch.cumsum(probs, dim=-1)
                mask = cumsum <= top_p
                mask[:, 0] = True  # Always include the highest logit
                values = values.masked_fill(~mask, float('-inf'))

            result = torch.full_like(logits, float('-inf'))
            result.scatter_(1, indices, values)
            return result

        # Only top-p sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 0] = False  # Keep at least one token
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        return logits.masked_fill(indices_to_remove, float('-inf'))

    @staticmethod
    def check_eos_vectorized(next_tokens: torch.Tensor, 
                              eos_token_ids: Optional[List[int]]) -> bool:
        """
        Check if all sequences have reached EOS using vectorized comparison.
        """
        if not eos_token_ids:
            return False

        eos_tensor = torch.tensor(eos_token_ids, device=next_tokens.device, dtype=next_tokens.dtype)
        matches = (next_tokens.unsqueeze(-1) == eos_tensor).any(dim=-1)
        return matches.all().item()

# ===============================
# Main LLM Class
# ===============================

class LLM(nn.Module):
    """
    Wrapper around GPT model with optimized text generation features.
    """
    def __init__(self, config: ModelConfig = None, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.config = config
        self.model = None
        self.sampler = OptimizedSampler()
        self.use_amp = torch.cuda.is_available() and device != "cpu"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        if config is not None:
            self.init_from_config(config)

    def init_from_config(self, config: ModelConfig, train_mode: bool = True):
        """Initialize model from config."""
        self.config = config
        self.model = GPT(config).to(self.device)
        self.model.train() if train_mode else self.model.eval()

    def load(self, model_variant: str, model_family: str = "gpt2", cache_dir: str = "./models"):
        """
        Load pre-trained weights for specified model variant.
        """
        if model_family != "gpt2":
            raise NotImplementedError(f"{model_family} not supported.")

        if self.model is None:
            if self.config is None:
                raise RuntimeError("Must set config before loading weights.")
            self.model = GPT(self.config).to(self.device)

        filename = f"model-{model_variant}.safetensors"
        url = get_gpt2_safetensors_url(model_variant)
        filepath = download_safetensors(filename, cache_dir, url)
        params = load_safetensors(filepath)

        with Spinner("Assigning weights to model"):
            self.model = load_gpt2_weights_meta(self.model, self.config, params, device=self.device)

    def save(self, save_path: str):
        """
        Save model weights to a file.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved to {save_path}")

    def _setup_generation(self, input_ids: torch.Tensor, generation_config: GenerationConfig):
        """
        Prepares model for generation: KV cache, initial forward, etc.
        """
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
        """
        Run a single forward step with optional mixed precision.
        """
        if use_mixed_precision and self.use_amp:
            with torch.cuda.amp.autocast():
                return self.model(input_token, use_cache=use_cache, pos_offset=pos_offset)
        return self.model(input_token, use_cache=use_cache, pos_offset=pos_offset)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, generation_config: GenerationConfig) -> Dict[str, Any]:
        """
        Run autoregressive generation from input tokens.
        """
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

            # Conditionally apply optimized sampling or naive fallback
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
            else:
                # Naive fallback: no repetition penalty or top-k/top-p filtering
                pass  # Keep logits as is or add custom naive sampling here

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1) if generation_config.do_sample else \
                         torch.argmax(probs, dim=-1, keepdim=True)

            # Optionally store logprobs
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
        """
        Generate text from a string prompt.
        """
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.generate(input_ids, generation_config)
        text = tokenizer.decode(output["tokens"][0], skip_special_tokens=True)

        return {
            "text": text,
            "tokens": output["tokens"][0].tolist() if generation_config.return_tokens else None,
            "logprobs": output.get("logprobs", None)
        }

    def generate_batch(self, prompts: List[str], tokenizer, generation_config: GenerationConfig) -> List[Dict[str, Any]]:
        """
        Generate multiple responses in batch mode.
        """
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

# ===============================
# Factory Function
# ===============================

def create_optimized_llm(model_variant: str = "gpt2", 
                         device: str = "auto",
                         cache_dir: str = "./models") -> LLM:
    """
    Factory method to create and load an LLM with default configuration.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    llm = LLM(device=device)
    # You can load pretrained weights here if needed:
    # llm.load(model_variant, cache_dir=cache_dir)
    return llm
