import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
import json
import os
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Import your existing modules
from config import Config
from model import GPT  # Assuming your GPT code is in paste.py

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLM:
    """
    A comprehensive wrapper class for Large Language Models.
    
    This class provides a unified interface for:
    - Model initialization from scratch
    - Loading pre-trained weights (HuggingFace or custom)
    - Text generation with various sampling strategies
    - Model management and configuration
    
    Features:
    - Support for custom architectures via Config
    - HuggingFace integration for popular models
    - Flexible text generation with multiple decoding strategies
    - KV caching for efficient inference
    - Device management and memory optimization
    - Model saving/loading utilities
    
    Usage:
        # Initialize from scratch
        llm = LLM(config)
        llm.initialize()
        
        # Load pre-trained model
        llm = LLM(config)
        llm.load(model_path_or_name)
        
        # Generate text
        output = llm.generate("Hello, world!", max_length=50)
    """
    
    def __init__(self, config: Union[Config, Dict[str, Any], str]):
        """
        Initialize the LLM wrapper.
        
        Args:
            config: Model configuration. Can be:
                - Config object
                - Dictionary with config parameters
                - Path to config JSON file
        """
        self.config = self._load_config(config)
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_initialized = False
        self.is_loaded = False
        
        logger.info(f"LLM wrapper initialized with device: {self.device}")
    
    def _load_config(self, config: Union[Config, Dict[str, Any], str]) -> Config:
        """Load configuration from various sources."""
        if isinstance(config, Config):
            return config
        elif isinstance(config, dict):
            return Config(**config)
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config_dict = json.load(f)
            return Config(**config_dict)
        else:
            raise ValueError("Config must be Config object, dict, or path to JSON file")
    
    def initialize(self, pretrained_weights: bool = False) -> 'LLM':
        """
        Initialize the model from scratch.
        
        Args:
            pretrained_weights: If True, initialize with pretrained-like weights
                              If False, use random initialization
        
        Returns:
            Self for method chaining
        """
        logger.info("Initializing model from scratch...")
        
        # Create model instance
        self.model = GPT(self.config)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Initialize weights
        if pretrained_weights:
            self._init_pretrained_weights()
        else:
            self._init_random_weights()
        
        # Initialize tokenizer if specified
        if hasattr(self.config, 'tokenizer_name') and self.config.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.is_initialized = True
        logger.info(f"Model initialized with {self._count_parameters():,} parameters")
        
        return self
    
    def load(self, 
             model_path_or_name: str, 
             from_hf: bool = True,
             merge_weights: bool = False,
             adapter_path: Optional[str] = None) -> 'LLM':
        """
        Load pre-trained model weights.
        
        Args:
            model_path_or_name: Path to model or HuggingFace model name
            from_hf: Whether to load from HuggingFace Hub
            merge_weights: Whether to merge adapter weights (for LoRA, etc.)
            adapter_path: Path to adapter weights if separate
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Loading model from: {model_path_or_name}")
        
        if from_hf:
            self._load_from_huggingface(model_path_or_name)
        else:
            self._load_from_checkpoint(model_path_or_name)
        
        if adapter_path and merge_weights:
            self._merge_adapter_weights(adapter_path)
        
        # Move to device
        if self.model:
            self.model = self.model.to(self.device)
        
        self.is_loaded = True
        logger.info("Model loaded successfully")
        
        return self
    
    def generate(self,
                prompt: Union[str, List[str]],
                max_length: int = 100,
                max_new_tokens: Optional[int] = None,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.9,
                do_sample: bool = True,
                num_return_sequences: int = 1,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[int] = None,
                use_cache: bool = True,
                repetition_penalty: float = 1.0,
                length_penalty: float = 1.0,
                early_stopping: bool = False,
                return_dict: bool = False) -> Union[str, List[str], Dict[str, Any]]:
        """
        Generate text using the loaded model.
        
        Args:
            prompt: Input text prompt(s)
            max_length: Maximum total sequence length
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to generate
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            use_cache: Whether to use KV caching
            repetition_penalty: Penalty for repeated tokens
            length_penalty: Length penalty for beam search
            early_stopping: Whether to stop early in beam search
            return_dict: Whether to return detailed generation info
        
        Returns:
            Generated text(s) or generation dictionary
        """
        if not (self.is_initialized or self.is_loaded):
            raise RuntimeError("Model must be initialized or loaded before generation")
        
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available. Set tokenizer_name in config or load HF model")
        
        # Handle batch prompts
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.block_size
        ).to(self.device)
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        # Calculate max_new_tokens if not provided
        if max_new_tokens is None:
            max_new_tokens = max_length - inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            if hasattr(self.model, 'generate') and hasattr(self.model, 'config'):
                # Use HuggingFace model's generate method
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=use_cache,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                    return_dict_in_generate=return_dict,
                    output_scores=return_dict
                )
            else:
                # Use custom generation method
                outputs = self._custom_generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    eos_token_id=eos_token_id,
                    use_cache=use_cache,
                    repetition_penalty=repetition_penalty
                )
        
        # Decode outputs
        if return_dict and hasattr(outputs, 'sequences'):
            sequences = outputs.sequences
        else:
            sequences = outputs if isinstance(outputs, torch.Tensor) else outputs.sequences
        
        # Remove input tokens from output
        if len(sequences.shape) > 1:
            new_tokens = sequences[:, inputs.input_ids.shape[1]:]
        else:
            new_tokens = sequences[inputs.input_ids.shape[1]:]
        
        # Decode to text
        generated_texts = self.tokenizer.batch_decode(
            new_tokens, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Format output
        if isinstance(prompt, str) and num_return_sequences == 1:
            result = generated_texts[0]
        else:
            result = generated_texts
        
        if return_dict:
            return {
                'generated_text': result,
                'sequences': sequences,
                'scores': getattr(outputs, 'scores', None),
                'input_ids': inputs.input_ids
            }
        
        return result
    
    def _custom_generate(self,
                        input_ids: torch.Tensor,
                        max_new_tokens: int = 50,
                        temperature: float = 1.0,
                        top_k: int = 50,
                        top_p: float = 0.9,
                        do_sample: bool = True,
                        num_return_sequences: int = 1,
                        eos_token_id: Optional[int] = None,
                        use_cache: bool = True,
                        repetition_penalty: float = 1.0) -> torch.Tensor:
        """Custom generation method for non-HF models."""
        batch_size, seq_len = input_ids.shape
        
        # Expand for multiple return sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            batch_size *= num_return_sequences
        
        # Initialize KV cache if using
        if use_cache and hasattr(self.model, 'initialize_kv_cache'):
            max_cache_len = seq_len + max_new_tokens
            self.model.initialize_kv_cache(
                batch_size, 
                max_cache_len,
                dtype=next(self.model.parameters()).dtype
            )
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get model output
            if use_cache and hasattr(self.model, 'kv_cache_initialized') and self.model.kv_cache_initialized:
                # Use only the last token for cached generation
                model_input = generated[:, -1:] if generated.shape[1] > seq_len else generated
                logits = self.model(model_input, use_cache=True)
            else:
                logits = self.model(generated, use_cache=False)
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply sampling
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Check for EOS token
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        
        return generated
    
    def _load_from_huggingface(self, model_name: str):
        """Load model from HuggingFace Hub."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                trust_remote_code=True
            )
            
        except Exception as e:
            logger.warning(f"Failed to load HF model directly: {e}")
            logger.info("Attempting to load weights into custom architecture...")
            
            # Initialize custom model
            self.model = GPT(self.config)
            
            # Load HF model for weight extraction
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Transfer weights
            self._transfer_hf_weights(hf_model)
    
    def _load_from_checkpoint(self, checkpoint_path: str):
        """Load model from local checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.is_file():
            # Single checkpoint file
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Initialize model if not done
            if self.model is None:
                self.model = GPT(self.config)
            
            # Load state dict
            self.model.load_state_dict(state_dict, strict=False)
            
        elif checkpoint_path.is_dir():
            # Directory with model files
            config_file = checkpoint_path / "config.json"
            model_file = checkpoint_path / "pytorch_model.bin"
            
            if not model_file.exists():
                model_file = checkpoint_path / "model.safetensors"
            
            if config_file.exists():
                with open(config_file) as f:
                    model_config = json.load(f)
                # Update config if needed
                
            if model_file.exists():
                if model_file.suffix == '.safetensors':
                    from safetensors.torch import load_file
                    state_dict = load_file(model_file)
                else:
                    state_dict = torch.load(model_file, map_location='cpu')
                
                # Initialize model if not done
                if self.model is None:
                    self.model = GPT(self.config)
                
                self.model.load_state_dict(state_dict, strict=False)
        
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    def _transfer_hf_weights(self, hf_model):
        """Transfer weights from HuggingFace model to custom model."""
        logger.info("Transferring weights from HuggingFace model...")
        
        hf_state_dict = hf_model.state_dict()
        custom_state_dict = self.model.state_dict()
        
        # Create mapping between HF and custom parameter names
        param_mapping = self._create_param_mapping(hf_state_dict, custom_state_dict)
        
        # Transfer weights
        for custom_name, hf_name in param_mapping.items():
            if hf_name in hf_state_dict and custom_name in custom_state_dict:
                hf_param = hf_state_dict[hf_name]
                custom_param = custom_state_dict[custom_name]
                
                if hf_param.shape == custom_param.shape:
                    custom_state_dict[custom_name].copy_(hf_param)
                    logger.debug(f"Transferred: {hf_name} -> {custom_name}")
                else:
                    logger.warning(f"Shape mismatch: {hf_name} {hf_param.shape} -> {custom_name} {custom_param.shape}")
        
        self.model.load_state_dict(custom_state_dict)
    
    def _create_param_mapping(self, hf_state_dict, custom_state_dict):
        """Create parameter name mapping between HF and custom models."""
        # This is a simplified mapping - you may need to customize based on specific architectures
        mapping = {}
        
        for custom_name in custom_state_dict.keys():
            # Try direct mapping first
            if custom_name in hf_state_dict:
                mapping[custom_name] = custom_name
                continue
            
            # Common transformations
            hf_name = custom_name
            
            # Handle transformer block naming
            hf_name = hf_name.replace('transformer.block_', 'transformer.h.')
            hf_name = hf_name.replace('.norm1.', '.ln_1.')
            hf_name = hf_name.replace('.norm2.', '.ln_2.')
            hf_name = hf_name.replace('.attn.qkv.', '.attn.c_attn.')
            hf_name = hf_name.replace('.attn.proj.', '.attn.c_proj.')
            hf_name = hf_name.replace('.mlp.fc.', '.mlp.c_fc.')
            hf_name = hf_name.replace('.mlp.proj.', '.mlp.c_proj.')
            
            if hf_name in hf_state_dict:
                mapping[custom_name] = hf_name
        
        return mapping
    
    def _merge_adapter_weights(self, adapter_path: str):
        """Merge adapter weights (e.g., LoRA) with base model."""
        logger.info(f"Merging adapter weights from: {adapter_path}")
        # Implementation depends on adapter type (LoRA, AdaLoRA, etc.)
        # This is a placeholder for adapter weight merging logic
        pass
    
    def _init_pretrained_weights(self):
        """Initialize weights similar to pretrained models."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _init_random_weights(self):
        """Initialize weights randomly."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.xavier_uniform_(module.weight)
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def save(self, save_path: str, save_tokenizer: bool = True):
        """Save model and configuration."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
        
        # Save config
        config_dict = self.config.__dict__ if hasattr(self.config, '__dict__') else vars(self.config)
        with open(save_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save tokenizer if available
        if self.tokenizer and save_tokenizer:
            self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to: {save_path}")
    
    def to(self, device: Union[str, torch.device]):
        """Move model to device."""
        self.device = torch.device(device)
        if self.model:
            self.model = self.model.to(self.device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        if self.model:
            self.model.eval()
        return self
    
    def train(self):
        """Set model to training mode."""
        if self.model:
            self.model.train()
        return self
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    
    def __repr__(self) -> str:
        status = []
        if self.is_initialized:
            status.append("initialized")
        if self.is_loaded:
            status.append("loaded")
        
        param_count = self._count_parameters() if self.model else 0
        
        return (f"LLM(config={type(self.config).__name__}, "
                f"parameters={param_count:,}, "
                f"device={self.device}, "
                f"status={','.join(status) if status else 'empty'})")


# Example usage and configuration
if __name__ == "__main__":
    # Example 1: Initialize from scratch
    config = Config.from_name("gpt2-small")
  
    
    # Example 2: Load from HuggingFace
    llm_hf = LLM(config)
    llm_hf.load("gpt2")
    
    # Example 3: Generate text
    output = llm_hf.generate(
        "The future of AI is",
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9
    )
    print(output)