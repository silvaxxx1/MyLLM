"""
LLM Wrapper for Memory-Optimized Loading and Generation

Provides a high-level interface for loading pretrained models and generating text
with support for various sampling strategies and optimization techniques.
"""

import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple

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
        self.tokenizer = None
        self.sampler = OptimizedSampler()
        self.use_amp = torch.cuda.is_available() and device != "cpu"
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        if config is not None:
            self.init_from_config(config)

    def init_from_config(self, config: ModelConfig, train_mode: bool = True) -> None:
        """Initialize GPT model structure without loading weights to GPU."""
        self.config = config
        self.model = GPT(config)
        
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        if not self.low_cpu_mem_usage:
            self.model.to(self.device, dtype=self.torch_dtype)

    def load(self, model_variant: str, model_family: Optional[str] = None) -> None:
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

    def list_models(self) -> List[str]:
        """List available cached models."""
        return self.loader.list_available_models()

    def save(self, save_path: str) -> None:
        """Save model weights to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")

    def _setup_generation(
        self, input_ids: torch.Tensor, generation_config: GenerationConfig
    ) -> Tuple[int, int, Optional[torch.Tensor]]:
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
                if generation_config.repetition_penalty != 1.0:
                    next_token_logits = self.sampler.apply_repetition_penalty_vectorized(
                        next_token_logits, generated, generation_config.repetition_penalty
                    )

                top_k = generation_config.top_k
                top_p = generation_config.top_p

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

    def __repr__(self) -> str:
        if self.model is None:
            return f"LLM(no model loaded, device='{self.device}')"
        name = getattr(self.config, 'name', 'unknown')
        n_params = sum(p.numel() for p in self.model.parameters())
        dtype = next(self.model.parameters()).dtype
        return f"LLM(model='{name}', params={n_params/1e6:.1f}M, device='{self.device}', dtype={dtype})"

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        low_cpu_mem_usage: bool = True,
    ) -> 'LLM':
        """
        One-line model + tokenizer loader.

        Args:
            model_name: e.g. 'gpt2-small', 'gpt2-medium'
            device: 'cuda' or 'cpu' (auto-detected if None)
            torch_dtype: optional dtype override (e.g. torch.float16)
            low_cpu_mem_usage: load weights incrementally to save memory

        Returns:
            Ready-to-use LLM with .tokenizer set.

        Example:
            llm = LLM.from_pretrained('gpt2-small')
            print(llm.generate_text('Hello world', GenerationConfig(max_length=50)))
        """
        from myllm.Tokenizers.factory import get_tokenizer

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        config = ModelConfig.from_name(model_name)
        llm = cls(config=config, device=device, torch_dtype=torch_dtype,
                  low_cpu_mem_usage=low_cpu_mem_usage)
        llm.load(model_name)

        # Map variant name → tokenizer family key
        FAMILY_MAP = {
            "gpt2": "gpt2", "gpt2-medium": "gpt2", "gpt2-large": "gpt2", "gpt2-xl": "gpt2",
            "llama2": "llama2", "llama3": "llama3",
            "mistral": "mistral",
            "gemma": "gemma",
            "phi": None,   # Phi-2 uses a custom CodeGen tokenizer; not auto-loaded
        }
        prefix = model_name.split('-')[0]
        family = FAMILY_MAP.get(prefix, prefix)

        if family is None:
            print(f"Note: '{model_name}' tokenizer is not auto-loadable. Pass one to generate_text().")
        else:
            try:
                # SentencePiece tokenizers need the .model file path
                SENTENCEPIECE_FAMILIES = {"llama2", "mistral", "gemma"}
                if family in SENTENCEPIECE_FAMILIES:
                    tok_path = llm.loader.download_tokenizer(model_name)
                    llm.tokenizer = get_tokenizer(family, model_path=tok_path)
                else:
                    llm.tokenizer = get_tokenizer(family)
            except Exception as e:
                print(f"Note: could not auto-load tokenizer for '{family}': {e}. Pass one to generate_text().")

        return llm

    def generate_text(
        self,
        prompt: str,
        tokenizer=None,
        generation_config: GenerationConfig = None,
        skip_prompt: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate text from string prompt.

        Args:
            prompt: Input text string
            tokenizer: Tokenizer instance (uses self.tokenizer if None)
            generation_config: Generation parameters (uses defaults if None)
            skip_prompt: If True, return only the newly generated tokens, not the prompt

        Returns:
            Dictionary containing generated text, tokens, and logprobs
        """
        tok = tokenizer if tokenizer is not None else self.tokenizer
        if tok is None:
            raise ValueError(
                "No tokenizer available. Pass one as the second argument "
                "or use LLM.from_pretrained() which auto-loads one."
            )
        if generation_config is None:
            generation_config = GenerationConfig()

        input_ids = tok.encode(prompt, return_tensors="pt").to(self.device)
        output = self.generate(input_ids, generation_config)

        tokens = output["tokens"][0]
        if skip_prompt:
            tokens = tokens[input_ids.shape[1]:]

        text = tok.decode(tokens, skip_special_tokens=True)

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