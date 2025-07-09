from dataclasses import dataclass
from typing import Optional, List

@dataclass
class GenerationConfig:
    # === Length Control ===
    max_length: int = 20  # Total maximum length (input + generated). Used when max_new_tokens is None.
    max_new_tokens: Optional[int] = None  # Max number of new tokens to generate (preferred over max_length).
    min_length: Optional[int] = None  # Minimum number of new tokens to generate before considering stopping.

    # === Sampling Strategies ===
    temperature: float = 1.0  # Controls randomness in sampling. Lower = more deterministic.
    top_k: Optional[int] = None  # Limit sampling to top-k most likely tokens (nucleus sampling).
    top_p: Optional[float] = None  # Limit sampling to top-p cumulative probability mass.
    typical_p: Optional[float] = None  # Typical decoding: favors tokens close to conditional median.

    do_sample: bool = True  # Whether to sample (True) or use greedy/beam search (False).

    # === KV Caching ===
    use_kv_cache: bool = True  # Enables use of key/value cache for faster generation during autoregressive decoding.

    # === Repetition Control ===
    repetition_penalty: float = 1.0  # >1 penalizes repeated tokens. Set >1.0 to reduce repetition.
    no_repeat_ngram_size: Optional[int] = None  # Prevents repeating any n-gram of given size in output.

    # === Stopping Criteria ===
    early_stopping: bool = True  # Stop generation when all beams reach EOS or constraints.

    eos_token_ids: Optional[List[int]] = None  # List of token IDs that indicate end-of-sequence.
    pad_token_id: Optional[int] = None  # Token ID used for padding sequences to equal length.

    # === Output Controls ===
    return_tokens: bool = True  # Whether to return generated token IDs.
    return_logprobs: bool = False  # Whether to return log probabilities of generated tokens.
    output_scores: bool = False  # Whether to return scores for generated sequences.
    output_attentions: bool = False  # Whether to return attention weights for each layer.
    output_hidden_states: bool = False  # Whether to return hidden states for each layer.
