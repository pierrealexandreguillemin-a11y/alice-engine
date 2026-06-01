"""Diversification - Hamming K-best post-MAP for Phase 4a.

Algorithm : greedy max-min Hamming-diversity (dispersion) heuristic. Iterate
candidates in descending score order; select a candidate iff its minimum Hamming
distance to every already-selected solution is >= min_hamming. This is the
greedy MaxMin-dispersion approach applied to diverse constraint solutions.

Reference : Hebrard, Hnich, O'Sullivan & Walsh (2005), "Finding Diverse and
Similar Solutions in Constraint Programming", Proc. AAAI'05, pp. 372-377.

For each candidate solution from MAP TopK, add to result iff min Hamming to
already-selected >= threshold. Degenerate inputs (pool size < K, empty list,
k=0) are handled gracefully without infinite loops.

ISO 5055 : SRP (diversification only, no MAP fit, no I/O).
ISO 24029 : Diversity stress test against single-mode solutions.
ISO 29119 : Deterministic ordering by score; testable pure functions.

Document ID: ALICE-ALI-DIVERSIFICATION
Version: 1.0.1
Count: K solutions per team per call
"""

from __future__ import annotations

from typing import Any

__all__ = ["SolutionTuple", "hamming_distance", "k_best_diversified"]

SolutionTuple = tuple[Any, ...]
"""A board-assignment solution tuple. Element type intentionally unparameterized
(heterogeneous: player ids or NR-FFE codes)."""


def hamming_distance(a: SolutionTuple, b: SolutionTuple) -> int:
    """Hamming distance between two tuples (count of unequal positions).

    Args:
    ----
        a: first tuple of arbitrary elements.
        b: second tuple of the same length as `a`.

    Returns:
    -------
        Number of positions where `a[i] != b[i]`.

    Raises:
    ------
        ValueError: if `len(a) != len(b)`.

    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(1 for x, y in zip(a, b, strict=True) if x != y)


def k_best_diversified(
    candidates: list[tuple[SolutionTuple, float]],
    k: int,
    min_hamming: int = 3,
) -> list[tuple[SolutionTuple, float]]:
    """Select K diversified solutions via greedy Hamming distance filter.

    Iterates through candidates in descending score order.  The highest-scoring
    candidate is always selected first.  Subsequent candidates are accepted iff
    their minimum Hamming distance to every already-selected solution is >=
    `min_hamming`.  Iteration stops as soon as `k` solutions are collected or
    all candidates are exhausted — no infinite loop on degenerate inputs.

    Args:
    ----
        candidates: list of (solution_tuple, score). May be unsorted.
        k: target number of diversified solutions to return.
        min_hamming: minimum Hamming distance between every pair in the result.
            Setting min_hamming=0 accepts all candidates up to k.

    Returns:
    -------
        List of at most `k` (solution_tuple, score) pairs in descending score
        order.  May be shorter than `k` when candidates are exhausted.

    """
    if not candidates:
        return []
    if k < 1:
        return []

    sorted_cands = sorted(candidates, key=lambda x: -x[1])
    selected: list[tuple[SolutionTuple, float]] = [sorted_cands[0]]

    for cand_sol, cand_score in sorted_cands[1:]:
        if len(selected) >= k:
            break
        min_h = min(hamming_distance(cand_sol, s) for s, _ in selected)
        if min_h >= min_hamming:
            selected.append((cand_sol, cand_score))

    return selected
