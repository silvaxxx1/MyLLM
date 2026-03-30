"""Tests for OptimizedSampler (myllm/utils/sampler.py)."""
import pytest
import torch
from myllm.utils.sampler import OptimizedSampler


@pytest.fixture(scope="module")
def sampler():
    return OptimizedSampler()


# ---------------------------------------------------------------------------
# Repetition penalty
# ---------------------------------------------------------------------------

class TestRepetitionPenalty:

    def test_penalises_generated_tokens(self, sampler):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        generated = torch.tensor([[1, 3]])          # tokens 1 and 3 were already generated
        result = sampler.apply_repetition_penalty_vectorized(logits, generated, penalty=2.0)
        assert result.shape == logits.shape
        # penalised tokens must be lower than originals
        assert result[0, 1] < logits[0, 1]
        assert result[0, 3] < logits[0, 3]

    def test_unpenalised_tokens_unchanged(self, sampler):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        generated = torch.tensor([[0]])             # only token 0 was generated
        result = sampler.apply_repetition_penalty_vectorized(logits, generated, penalty=2.0)
        assert torch.isclose(result[0, 1], logits[0, 1])
        assert torch.isclose(result[0, 2], logits[0, 2])

    def test_penalty_1_is_identity(self, sampler):
        logits = torch.randn(1, 100)
        generated = torch.randint(0, 100, (1, 20))
        result = sampler.apply_repetition_penalty_vectorized(logits, generated, penalty=1.0)
        assert torch.allclose(result, logits)

    def test_output_shape_preserved(self, sampler):
        logits = torch.randn(2, 200)
        generated = torch.randint(0, 200, (2, 10))
        result = sampler.apply_repetition_penalty_vectorized(logits, generated, penalty=1.3)
        assert result.shape == logits.shape


# ---------------------------------------------------------------------------
# Top-K sampling
# ---------------------------------------------------------------------------

class TestTopKSampling:

    def test_top_k_1_keeps_only_argmax(self, sampler):
        logits = torch.randn(1, 100)
        best_idx = logits.argmax(dim=-1).item()
        result = sampler.combined_top_k_top_p_sampling(logits, top_k=1)
        assert result.argmax(dim=-1).item() == best_idx

    def test_top_k_filters_to_k_candidates(self, sampler):
        logits = torch.randn(1, 1000)
        k = 50
        result = sampler.combined_top_k_top_p_sampling(logits, top_k=k)
        finite = torch.isfinite(result[0])
        assert finite.sum().item() <= k

    def test_output_shape_preserved(self, sampler):
        logits = torch.randn(2, 500)
        result = sampler.combined_top_k_top_p_sampling(logits, top_k=50)
        assert result.shape == logits.shape

    def test_top_k_no_nans(self, sampler):
        logits = torch.randn(1, 100)
        result = sampler.combined_top_k_top_p_sampling(logits, top_k=10)
        finite = result[torch.isfinite(result)]
        assert not torch.isnan(finite).any()


# ---------------------------------------------------------------------------
# Top-P sampling
# ---------------------------------------------------------------------------

class TestTopPSampling:

    def test_output_shape_preserved(self, sampler):
        logits = torch.randn(1, 1000)
        result = sampler.combined_top_k_top_p_sampling(logits, top_p=0.9)
        assert result.shape == logits.shape

    def test_top_p_1_keeps_all_tokens(self, sampler):
        logits = torch.randn(1, 100)
        result = sampler.combined_top_k_top_p_sampling(logits, top_p=1.0)
        # With p=1.0 no token should be filtered
        assert torch.isfinite(result).all()

    def test_combined_top_k_and_top_p(self, sampler):
        logits = torch.randn(1, 1000)
        result = sampler.combined_top_k_top_p_sampling(logits, top_k=100, top_p=0.9)
        assert result.shape == logits.shape


# ---------------------------------------------------------------------------
# EOS detection
# ---------------------------------------------------------------------------

class TestEOSDetection:

    def test_all_eos_returns_true(self, sampler):
        tokens = torch.tensor([[2]])
        assert sampler.check_eos_vectorized(tokens, eos_token_ids=[2]) is True

    def test_no_eos_returns_false(self, sampler):
        tokens = torch.tensor([[5]])
        assert sampler.check_eos_vectorized(tokens, eos_token_ids=[2]) is False

    def test_none_eos_returns_false(self, sampler):
        tokens = torch.tensor([[2]])
        assert sampler.check_eos_vectorized(tokens, eos_token_ids=None) is False

    def test_partial_batch_eos_returns_false(self, sampler):
        # Only some sequences hit EOS — all must hit EOS to return True
        tokens = torch.tensor([[2], [5]])
        assert sampler.check_eos_vectorized(tokens, eos_token_ids=[2]) is False
