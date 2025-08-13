
import torch
from typing import Optional, List


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

        batch_size, _ = logits.shape
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
