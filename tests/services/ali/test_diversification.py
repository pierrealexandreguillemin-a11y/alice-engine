"""Tests for services.ali.diversification (Phase 4a T3).

ISO 5055  : SRP per test class.
ISO 24029 : Robustness cases — degenerate pool < K, empty, k=0, identical.
ISO 29119 : >= 8 test cases, deterministic, documented.

Tests cover:
  - TestHammingDistance    : zero, partial, full, length-mismatch error.
  - TestKBestDiversified   : filter, k>pool, empty, k=0, score ordering,
                             determinism, large pool sanity.

Document ID: ALICE-TEST-ALI-DIVERSIFICATION
Version: 1.0.0
Count: 15 test cases
"""

from __future__ import annotations

import pytest

from services.ali.diversification import hamming_distance, k_best_diversified


# ---------------------------------------------------------------------------
# Hamming distance
# ---------------------------------------------------------------------------
class TestHammingDistance:
    """Unit tests for hamming_distance pure function."""

    def test_equal_tuples_distance_zero(self) -> None:
        assert hamming_distance((1, 2, 3), (1, 2, 3)) == 0

    def test_all_different(self) -> None:
        assert hamming_distance(("a", "b", "c"), ("x", "y", "z")) == 3

    def test_partial_difference(self) -> None:
        assert hamming_distance((1, 2, 3, 4), (1, 9, 3, 9)) == 2

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            hamming_distance((1, 2), (1, 2, 3))


# ---------------------------------------------------------------------------
# K-best diversified
# ---------------------------------------------------------------------------
class TestKBestDiversified:
    """Tests for k_best_diversified greedy selection."""

    def test_trivial_three_diverse(self) -> None:
        cands: list[tuple[tuple[int, ...], float]] = [
            ((1, 2, 3, 4, 5), 1.0),
            ((6, 7, 8, 9, 10), 0.9),
            ((11, 12, 13, 14, 15), 0.8),
        ]
        result = k_best_diversified(cands, k=3, min_hamming=3)
        assert len(result) == 3

    def test_hamming_threshold_filters_near_duplicates(self) -> None:
        cands: list[tuple[tuple[int, ...], float]] = [
            ((1, 2, 3, 4, 5), 1.0),
            ((1, 2, 3, 4, 9), 0.95),  # hamming=1 vs first -> rejected
            ((1, 2, 9, 9, 9), 0.9),  # hamming=3 vs first -> accepted
        ]
        result = k_best_diversified(cands, k=3, min_hamming=3)
        assert len(result) == 2  # first + third (second too close to first)
        assert result[0][1] == 1.0
        assert result[1][1] == 0.9

    def test_k_larger_than_diverse_candidates(self) -> None:
        """Degenerate pool : pool size < K, no infinite loop (ISO 24029)."""
        cands: list[tuple[tuple[int, ...], float]] = [
            ((1, 2, 3), 1.0),
            ((1, 2, 3), 0.9),  # identical to first -> rejected at min_hamming=1
        ]
        result = k_best_diversified(cands, k=5, min_hamming=1)
        assert len(result) == 1  # second is identical to first

    def test_empty_candidates(self) -> None:
        assert k_best_diversified([], k=5) == []

    def test_k_zero(self) -> None:
        result = k_best_diversified([((1, 2), 1.0)], k=0)
        assert result == []

    def test_descending_score_order(self) -> None:
        """Result must be ordered by descending score."""
        cands: list[tuple[tuple[int, ...], float]] = [
            ((1, 2, 3), 0.5),
            ((4, 5, 6), 0.9),
            ((7, 8, 9), 0.7),
        ]
        result = k_best_diversified(cands, k=3, min_hamming=1)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_deterministic_same_input_same_output(self) -> None:
        cands: list[tuple[tuple[int, ...], float]] = [
            ((1, 2, 3), 1.0),
            ((4, 5, 6), 0.9),
            ((7, 8, 9), 0.8),
        ]
        r1 = k_best_diversified(cands, k=2)
        r2 = k_best_diversified(cands, k=2)
        assert r1 == r2

    def test_pool_smaller_than_k_returns_fewer(self) -> None:
        """Explicit degenerate case : 3 distinct cands, k=10 returns 3."""
        cands: list[tuple[tuple[int, ...], float]] = [
            ((1, 0, 0), 1.0),
            ((0, 1, 0), 0.8),
            ((0, 0, 1), 0.6),
        ]
        result = k_best_diversified(cands, k=10, min_hamming=2)
        assert len(result) == 3

    def test_large_pool_k10_hamming3(self) -> None:
        """Sanity: K=10 diversity on N=20 distinct tuples returns exactly 10."""
        cands: list[tuple[tuple[int, ...], float]] = [
            (tuple(range(i * 5, i * 5 + 5)), float(20 - i)) for i in range(20)
        ]
        result = k_best_diversified(cands, k=10, min_hamming=3)
        assert len(result) == 10
        # Verify every pair meets the Hamming constraint
        for i, (s_i, _) in enumerate(result):
            for j, (s_j, _) in enumerate(result):
                if i != j:
                    assert hamming_distance(s_i, s_j) >= 3

    def test_min_hamming_zero_accepts_all_up_to_k(self) -> None:
        """min_hamming=0 : all candidates (even identical) accepted up to k."""
        cands: list[tuple[tuple[int, ...], float]] = [
            ((1, 2, 3), 1.0),
            ((1, 2, 3), 0.9),
            ((1, 2, 3), 0.8),
        ]
        result = k_best_diversified(cands, k=2, min_hamming=0)
        assert len(result) == 2

    def test_k_one_returns_highest_score_only(self) -> None:
        cands: list[tuple[tuple[int, ...], float]] = [
            ((1, 2, 3), 0.5),
            ((4, 5, 6), 0.9),
        ]
        result = k_best_diversified(cands, k=1, min_hamming=5)
        assert len(result) == 1
        assert result[0][1] == 0.9
