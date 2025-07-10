from dataclasses import dataclass 
from typing import Optional, List


# ===============================
# Configuration for Text Generation
# ===============================

@dataclass
class GenerationConfig:
    """
    Configuration dataclass for controlling the generation behavior of LLMs.
    """
    max_length: int = 20  # Total number of tokens to generate
    max_new_tokens: Optional[int] = None  # Alias for max_length override
    min_length: Optional[int] = None  # Minimum number of tokens to generate
    temperature: float = 1.0  # Softmax temperature (higher = more random)
    top_k: Optional[int] = None  # Top-k sampling
    top_p: Optional[float] = None  # Top-p (nucleus) sampling
    typical_p: Optional[float] = None  # Typical sampling (not used here)
    do_sample: bool = True  # Whether to sample or do greedy decoding
    use_kv_cache: bool = True  # Whether to use KV caching
    repetition_penalty: float = 1.0  # Penalize repeated tokens
    apply_repetition_penalty: bool = True  # Toggle repetition penalty on/off
    apply_top_k_sampling: bool = True  # Toggle top-k sampling on/off
    apply_top_p_sampling: bool = True  # Toggle top-p sampling on/off
    no_repeat_ngram_size: Optional[int] = None  # Prevent n-gram repetitions (not used)
    early_stopping: bool = True  # Stop generation on EOS
    eos_token_ids: Optional[List[int]] = None  # End-of-sequence token(s)
    pad_token_id: Optional[int] = None  # Padding token
    return_tokens: bool = True  # Return token IDs in output
    return_logprobs: bool = False  # Return log-probabilities
    output_scores: bool = False  # Output scores (not used)
    output_attentions: bool = False  # Output attentions (not used)
    output_hidden_states: bool = False  # Output hidden states (not used)
    use_mixed_precision: bool = True  # Use FP16 inference if possible
    max_logprob_history: int = 100  # Keep last N logprobs
    batch_size: int = 1  # For batched generation
    use_optimized_sampler: bool = True  # Whether to apply optimized sampling
