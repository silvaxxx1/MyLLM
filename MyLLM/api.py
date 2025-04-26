"""
API Wrapper for GPT-Style Transformer Model

This module provides a user-friendly interface for interacting with the decoder-only
transformer model, handling all the complexity of tokenization, generation strategies,
and memory management.

Key Features:
- Clean, high-level API for text generation
- Support for multiple generation strategies:
  * Greedy decoding
  * Top-k sampling
  * Top-p (nucleus) sampling
  * Temperature-controlled sampling
- Automatic KV cache management
- Batch processing support
- Streaming output option
- Configurable generation parameters

Example Usage:
    >>> from api import TextGenerator
    >>> generator = TextGenerator.from_pretrained("llama2-7b")
    >>> output = generator.generate("The future of AI is")
    >>> print(output)
"""

import torch
from typing import List, Optional, Union, Dict, Any
from transformers import AutoTokenizer

class TextGenerator:
    """
    High-level API for text generation with GPT-style models.
    
    Handles all aspects of text generation including:
    - Tokenization
    - Generation strategies
    - Memory management
    - Output decoding
    
    Args:
        model: Loaded GPT model instance
        tokenizer: Pre-trained tokenizer
        device: Device to run the model on ('cuda' or 'cpu')
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.default_generation_params = {
            'max_length': 100,
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.2,
            'do_sample': True,
        }
        
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Load a pre-trained model and tokenizer.
        
        Args:
            model_name_or_path: Name of pre-trained model or path to local checkpoint
            **kwargs: Additional arguments for model loading
            
        Returns:
            TextGenerator instance
        """
        # In a real implementation, you would load your model here
        # This is just a placeholder for the concept
        config = Config.from_name(model_name_or_path)
        model = GPT(config)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, **kwargs)
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_return_sequences: int = 1,
        stream: bool = False,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text or list of texts to complete
            max_length: Maximum length of generated text
            temperature: Controls randomness (lower = more deterministic)
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability threshold for nucleus sampling
            repetition_penalty: Penalty for repeated tokens (>1 discourages repetition)
            do_sample: Whether to use sampling instead of greedy decoding
            num_return_sequences: Number of sequences to generate per input
            stream: Whether to yield tokens as they're generated
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text or list of generated texts
        """
        # Merge provided params with defaults
        generation_params = self._get_generation_params(
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            **kwargs
        )
        
        # Tokenize input
        input_ids = self._tokenize_prompt(prompt, num_return_sequences)
        
        # Prepare model for generation
        self.model.eval()
        self._prepare_for_generation(input_ids)
        
        # Generate text
        if stream:
            return self._stream_generation(input_ids, generation_params)
        else:
            return self._batch_generate(input_ids, generation_params)
    
    def _get_generation_params(self, **kwargs) -> Dict[str, Any]:
        """Merge provided generation parameters with defaults."""
        params = self.default_generation_params.copy()
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return params
    
    def _tokenize_prompt(self, prompt: Union[str, List[str]], num_return_sequences: int) -> torch.Tensor:
        """Tokenize prompt and prepare for model input."""
        if isinstance(prompt, str):
            prompt = [prompt]
            
        encodings = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.block_size
        ).to(self.device)
        
        # Expand for multiple sequences if needed
        if num_return_sequences > 1:
            encodings = {k: v.repeat_interleave(num_return_sequences, dim=0) 
                        for k, v in encodings.items()}
            
        return encodings.input_ids
    
    def _prepare_for_generation(self, input_ids: torch.Tensor):
        """Initialize KV cache and other generation setup."""
        batch_size = input_ids.size(0)
        max_length = self.model.config.block_size
        
        if not self.model.kv_cache_initialized:
            self.model.initialize_kv_cache(
                batch_size=batch_size,
                max_seq_len=max_length,
                dtype=next(self.model.parameters()).dtype
            )
        else:
            self.model.reset_cache()
    
    def _batch_generate(self, input_ids: torch.Tensor, generation_params: Dict[str, Any]) -> Union[str, List[str]]:
        """Generate text in batch mode."""
        with torch.no_grad():
            outputs = self._generate_sequences(input_ids, generation_params)
        
        return self._decode_outputs(outputs, input_ids.size(0))
    
    def _stream_generation(self, input_ids: torch.Tensor, generation_params: Dict[str, Any]):
        """Generate text with streaming output."""
        # This would be implemented with a generator that yields tokens
        # as they're generated, but omitted for brevity
        raise NotImplementedError("Streaming generation not implemented in this example")
    
    def _generate_sequences(self, input_ids: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Core generation logic."""
        batch_size = input_ids.size(0)
        current_input = input_ids
        generated = input_ids.clone()
        
        for _ in range(params['max_length']):
            # Get model outputs
            logits = self.model(current_input, use_cache=True)
            
            # Apply repetition penalty if specified
            if params['repetition_penalty'] != 1.0:
                logits = self._apply_repetition_penalty(logits, generated, params['repetition_penalty'])
            
            # Get next token
            next_token = self._get_next_token(logits, params)
            
            # Append to generated sequences
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Update input for next step
            current_input = next_token
            
            # Early stopping if all sequences hit EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break
                
        return generated
    
    def _get_next_token(self, logits: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Select next token based on generation parameters."""
        # Apply temperature
        if params['temperature'] != 1.0:
            logits = logits / params['temperature']
            
        # Apply top-k filtering
        if params['top_k'] > 0:
            logits = self._top_k_filtering(logits, params['top_k'])
            
        # Apply top-p filtering
        if params['top_p'] > 0:
            logits = self._top_p_filtering(logits, params['top_p'])
            
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        if params['do_sample']:
            # Sample from distribution
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
        return next_token
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filter logits to only the top k options."""
        values, indices = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        return torch.where(logits < min_values, torch.ones_like(logits) * -float('Inf'), logits)
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus (top-p) filtering of logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Create mask and scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('Inf')
        return logits
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
        """Apply repetition penalty to discourage repeated tokens."""
        for i in range(generated.size(0)):
            for token in generated[i].unique():
                if logits[i, token] < 0:
                    logits[i, token] *= penalty
                else:
                    logits[i, token] /= penalty
        return logits
    
    def _decode_outputs(self, outputs: torch.Tensor, num_inputs: int) -> Union[str, List[str]]:
        """Decode generated token ids to text."""
        sequences = []
        for i, seq in enumerate(outputs):
            # Remove input prompt and special tokens
            text = self.tokenizer.decode(
                seq[len(self.tokenizer.encode(self.tokenizer.bos_token)):], 
                skip_special_tokens=True
            )
            
            # Group multiple sequences per input
            if i < num_inputs:
                sequences.append([text])
            else:
                sequences[i % num_inputs].append(text)
                
        # Flatten if single input/single sequence
        if len(sequences) == 1 and len(sequences[0]) == 1:
            return sequences[0][0]
        elif all(len(s) == 1 for s in sequences):
            return [s[0] for s in sequences]
        else:
            return sequences