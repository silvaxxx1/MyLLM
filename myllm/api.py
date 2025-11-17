"""
LLM Wrapper for Memory-Optimized Loading and Generation

Provides a high-level interface for loading pretrained models and generating text
with support for various sampling strategies and optimization techniques.
"""

import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

from myllm.model import GPT
from myllm.Configs import ModelConfig, GenerationConfig
from myllm.utils import OptimizedSampler
from myllm.utils.loader import ModelLoader
from myllm.utils import WEIGHT_MAPPERS


class LLM(nn.Module):
    """
    High-level wrapper around GPT models with memory-optimized loading and generation.
    
    Features:
    - Automatic model downloading and caching
    - Memory-efficient loading (low_cpu_mem_usage mode)
    - KV caching for fast generation
    - Multiple sampling strategies (top-k, top-p, temperature)
    - Batch generation support
    - Mixed precision inference
    """
    
    def __init__(
        self,
        config: ModelConfig = None,
        torch_dtype: Optional[torch.dtype] = None,
        low_cpu_mem_usage: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.model: Optional[nn.Module] = None
        self.loader = ModelLoader(cache_dir="./models")
        self.torch_dtype = torch_dtype or torch.float32
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.sampler = OptimizedSampler()
        self.use_amp = torch.cuda.is_available() and device != "cpu"
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        if config is not None:
            self.init_from_config(config)

    def init_from_config(self, config: ModelConfig, train_mode: bool = True):
        """Initialize GPT model structure without loading weights to GPU."""
        self.config = config
        self.model = GPT(config)
        
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        if not self.low_cpu_mem_usage:
            self.model.to(self.device, dtype=self.torch_dtype)

    def load(self, model_variant: str, model_family: Optional[str] = None):
        """
        Load pretrained model weights.
        
        Args:
            model_variant: Model name (e.g., "gpt2-small", "llama2-7b")
            model_family: Model family (auto-detected if None)
        """
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
        """List available cached models."""
        return self.loader.list_available_models()

    def save(self, save_path: str):
        """Save model weights to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")

    def _setup_generation(
        self, input_ids: torch.Tensor, generation_config: GenerationConfig
    ):
        """Initialize generation, set up KV cache, and process prompt."""
        B, T = input_ids.shape
        
        if generation_config.use_kv_cache:
            # Initialize KV cache
            self.model.reset_cache()
            max_seq_len = T + generation_config.max_length
            self.model.initialize_kv_cache(
                batch_size=B,
                max_seq_len=max_seq_len,
                dtype=self.torch_dtype
            )
            
            # Process prompt through model with cache
            with torch.no_grad():
                if generation_config.use_mixed_precision and self.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(input_ids, use_cache=True, pos_offset=0)
                else:
                    logits = self.model(input_ids, use_cache=True, pos_offset=0)
            
            return B, T, logits
        
        return B, T, None

    def _forward_step(
        self, input_token: torch.Tensor, use_cache: bool, 
        pos_offset: int, use_mixed_precision: bool
    ) -> torch.Tensor:
        """Execute single forward pass through model."""
        with torch.no_grad():
            if use_mixed_precision and self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits = self.model(
                        input_token, use_cache=use_cache, pos_offset=pos_offset
                    )
            else:
                logits = self.model(
                    input_token, use_cache=use_cache, pos_offset=pos_offset
                )
        return logits

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, generation_config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Generate text autoregressively from input prompt.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            generation_config: Generation parameters
            
        Returns:
            Dictionary containing:
                - tokens: Generated token IDs
                - logprobs: Log probabilities (if return_logprobs=True)
        """
        self.model.eval()
        
        # Setup and get initial logits from cached prompt
        B, T, initial_logits = self._setup_generation(input_ids, generation_config)
        generated = input_ids.clone()
        logprobs = []
        max_logprob_history = generation_config.max_logprob_history
        
        # EOS token handling
        eos_tensor = (
            torch.tensor(generation_config.eos_token_ids, device=input_ids.device) 
            if generation_config.eos_token_ids else None
        )
        
        # Generation loop
        for step in range(generation_config.max_length):
            current_seq_len = generated.shape[1]

            # Use cached logits for first prediction, forward pass for rest
            if step == 0 and generation_config.use_kv_cache and initial_logits is not None:
                logits = initial_logits
            else:
                # Process last added token
                if generation_config.use_kv_cache:
                    input_token = generated[:, -1:]
                    pos_offset = current_seq_len - 1
                else:
                    input_token = generated
                    pos_offset = 0

                logits = self._forward_step(
                    input_token, generation_config.use_kv_cache,
                    pos_offset, generation_config.use_mixed_precision
                )

            # Sample next token
            next_token_logits = logits[:, -1, :] / generation_config.temperature

            # Apply sampling strategies
            if generation_config.use_optimized_sampler:
                if (generation_config.apply_repetition_penalty and 
                    generation_config.repetition_penalty != 1.0):
                    next_token_logits = self.sampler.apply_repetition_penalty_vectorized(
                        next_token_logits, generated, generation_config.repetition_penalty
                    )

                top_k = generation_config.top_k if generation_config.apply_top_k_sampling else None
                top_p = generation_config.top_p if generation_config.apply_top_p_sampling else None

                next_token_logits = self.sampler.combined_top_k_top_p_sampling(
                    next_token_logits, top_k, top_p
                )

            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample token
            if generation_config.do_sample:
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # Store logprobs if requested
            if generation_config.return_logprobs:
                logprob = torch.log(probs.gather(1, next_token))
                logprobs.append(logprob)
                if len(logprobs) > max_logprob_history:
                    logprobs.pop(0)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for repetition (simple heuristic)
            if step > 10 and step % 5 == 0:
                last_5_tokens = generated[0, -5:].tolist()
                if len(set(last_5_tokens)) == 1:
                    break  # Stop if repeating same token

            # Check for EOS
            if generation_config.early_stopping and eos_tensor is not None:
                if self.sampler.check_eos_vectorized(
                    next_token, generation_config.eos_token_ids
                ):
                    break

        # Return results
        output = {"tokens": generated}
        if generation_config.return_logprobs and logprobs:
            output["logprobs"] = torch.cat(logprobs, dim=1)
        return output

    def generate_text(
        self, prompt: str, tokenizer, generation_config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Generate text from string prompt.
        
        Args:
            prompt: Input text string
            tokenizer: Tokenizer instance
            generation_config: Generation parameters
            
        Returns:
            Dictionary containing generated text, tokens, and logprobs
        """
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.generate(input_ids, generation_config)
        text = tokenizer.decode(output["tokens"][0], skip_special_tokens=True)
        
        return {
            "text": text,
            "tokens": (output["tokens"][0].tolist() 
                      if generation_config.return_tokens else None),
            "logprobs": output.get("logprobs", None)
        }

    def generate_batch(
        self, prompts: List[str], tokenizer, generation_config: GenerationConfig
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input text strings
            tokenizer: Tokenizer instance
            generation_config: Generation parameters
            
        Returns:
            List of dictionaries with generated text, tokens, and logprobs
        """
        if len(prompts) == 0:
            return []

        # Encode and pad prompts
        encoded = [tokenizer.encode(p, return_tensors="pt") for p in prompts]
        max_len = max(e.shape[1] for e in encoded)

        padded = []
        for e in encoded:
            pad_len = max_len - e.shape[1]
            pad_token_id = (generation_config.pad_token_id or 
                           tokenizer.pad_token_id or 0)
            if pad_len > 0:
                padding = torch.full((1, pad_len), pad_token_id, dtype=e.dtype)
                e = torch.cat([e, padding], dim=1)
            padded.append(e)

        # Generate
        batch_input = torch.cat(padded, dim=0).to(self.device)
        batch_output = self.generate(batch_input, generation_config)

        # Decode results
        results = []
        for i in range(len(prompts)):
            tokens = batch_output["tokens"][i]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            result = {
                "text": text,
                "tokens": (tokens.tolist() 
                          if generation_config.return_tokens else None),
                "logprobs": (batch_output.get("logprobs", [None] * len(prompts))[i] 
                            if generation_config.return_logprobs else None)
            }
            results.append(result)

        return results