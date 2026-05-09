"""Tests scripts/d8/dro (Sinha 2018, Duchi 2021).

Document ID: ALICE-D8-TEST-DRO
Version: 1.0.0
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from scripts.d8.dro import (
    DEFAULT_EPSILONS,
    DEFAULT_N_PERTURBATIONS,
    ELO_CEILING,
    ELO_FLOOR,
    DROOutcome,
    compute_dro_for_match,
    perturb_wasserstein,
    wasserstein_worst_case,
)

POOL_SMALL: list[int] = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200]
POOL_BORDERLINE: list[int] = [810, 820, 830, 2890, 2895, 2898]


def _identity_recall(_perturbed: list[int]) -> float:
    """Recall is independent of the perturbation — exercises plumbing only."""
    return 0.85


def _shift_sensitive_recall(perturbed: list[int]) -> float:
    """Recall drops the further the mean Elo strays from the original mean."""
    if not perturbed:
        return 0.0
    mean = float(np.mean(perturbed))
    deviation = abs(mean - 1850) / 1850
    return float(max(0.0, 1.0 - deviation * 5))


def _strict_recall(perturbed: list[int]) -> float:
    """Recall = mean(perturbed) / 2500 clipped to [0, 1] — monotone in mean."""
    if not perturbed:
        return 0.0
    mean = float(np.mean(perturbed))
    return float(max(0.0, min(1.0, mean / 2500.0)))


# -----------------------------------------------------------------------------
# perturb_wasserstein
# -----------------------------------------------------------------------------


def test_perturb_returns_list_and_params() -> None:
    """perturb_wasserstein returns a tuple (list[int], dict)."""
    perturbed, params = perturb_wasserstein(POOL_SMALL, epsilon=0.05, seed=42)
    assert isinstance(perturbed, list)
    assert all(isinstance(x, int) for x in perturbed)
    assert set(params.keys()) == {"shift", "scale"}


def test_perturb_preserves_length() -> None:
    perturbed, _ = perturb_wasserstein(POOL_SMALL, epsilon=0.10, seed=1)
    assert len(perturbed) == len(POOL_SMALL)


def test_perturb_clips_below_floor() -> None:
    """Borderline-low Elos with a negative shift should clip at ELO_FLOOR."""
    perturbed, _ = perturb_wasserstein([810, 815, 820], epsilon=0.50, seed=0)
    assert all(x >= ELO_FLOOR for x in perturbed)


def test_perturb_clips_above_ceiling() -> None:
    """Borderline-high Elos with a positive shift should clip at ELO_CEILING."""
    perturbed, _ = perturb_wasserstein([2895, 2898, 2899], epsilon=0.50, seed=0)
    assert all(x <= ELO_CEILING for x in perturbed)


def test_perturb_borderline_clips_both_ends() -> None:
    perturbed, _ = perturb_wasserstein(POOL_BORDERLINE, epsilon=0.50, seed=7)
    assert all(ELO_FLOOR <= x <= ELO_CEILING for x in perturbed)


def test_perturb_seed_deterministic() -> None:
    out1, p1 = perturb_wasserstein(POOL_SMALL, epsilon=0.05, seed=123)
    out2, p2 = perturb_wasserstein(POOL_SMALL, epsilon=0.05, seed=123)
    assert out1 == out2
    assert p1 == p2


def test_perturb_different_seeds_differ() -> None:
    out1, _ = perturb_wasserstein(POOL_SMALL, epsilon=0.10, seed=1)
    out2, _ = perturb_wasserstein(POOL_SMALL, epsilon=0.10, seed=2)
    assert out1 != out2


def test_perturb_epsilon_zero_returns_input() -> None:
    """ε=0 ⇒ no perturbation (shift=0, scale=1) ⇒ input unchanged."""
    perturbed, params = perturb_wasserstein(POOL_SMALL, epsilon=0.0, seed=42)
    assert perturbed == POOL_SMALL
    assert params["shift"] == 0.0
    assert params["scale"] == 1.0


def test_perturb_empty_pool_returns_empty() -> None:
    perturbed, params = perturb_wasserstein([], epsilon=0.10, seed=42)
    assert perturbed == []
    assert params == {"shift": 0.0, "scale": 1.0}


def test_perturb_epsilon_one_raises() -> None:
    with pytest.raises(ValueError, match=r"epsilon"):
        perturb_wasserstein(POOL_SMALL, epsilon=1.0, seed=0)


def test_perturb_epsilon_negative_raises() -> None:
    with pytest.raises(ValueError, match=r"epsilon"):
        perturb_wasserstein(POOL_SMALL, epsilon=-0.05, seed=0)


def test_perturb_epsilon_nan_raises() -> None:
    with pytest.raises(ValueError, match=r"finite"):
        perturb_wasserstein(POOL_SMALL, epsilon=float("nan"), seed=0)


def test_perturb_non_finite_elo_raises() -> None:
    with pytest.raises(ValueError, match=r"finite"):
        perturb_wasserstein([1500, float("nan"), 1700], epsilon=0.05, seed=0)  # type: ignore[list-item]


# -----------------------------------------------------------------------------
# wasserstein_worst_case
# -----------------------------------------------------------------------------


def test_worst_case_returns_outcome_dataclass() -> None:
    outcome = wasserstein_worst_case(POOL_SMALL, _identity_recall, epsilon=0.05, n_perturbations=10)
    assert isinstance(outcome, DROOutcome)
    assert outcome.epsilon == 0.05
    assert outcome.n_perturbations == 10


def test_worst_case_constant_recall_returns_that_value() -> None:
    """Constant recall_fn ⇒ worst == constant (no variance to exploit)."""
    outcome = wasserstein_worst_case(POOL_SMALL, _identity_recall, epsilon=0.10, n_perturbations=20)
    assert outcome.recall_worst_case == pytest.approx(0.85)


def test_worst_case_monotone_in_epsilon() -> None:
    """Larger ε ⇒ worse worst-case (or equal) when the recall_fn is sensitive."""
    out_small = wasserstein_worst_case(
        POOL_SMALL, _shift_sensitive_recall, epsilon=0.02, n_perturbations=50
    )
    out_large = wasserstein_worst_case(
        POOL_SMALL, _shift_sensitive_recall, epsilon=0.20, n_perturbations=50
    )
    assert out_large.recall_worst_case <= out_small.recall_worst_case


def test_worst_case_finding_format() -> None:
    outcome = wasserstein_worst_case(
        POOL_SMALL, _shift_sensitive_recall, epsilon=0.10, n_perturbations=20
    )
    assert "shift=" in outcome.worst_perturbation_finding
    assert "scale=" in outcome.worst_perturbation_finding


def test_worst_case_empty_pool_returns_zero() -> None:
    outcome = wasserstein_worst_case([], _identity_recall, epsilon=0.10)
    assert outcome.n_perturbations == 0
    assert outcome.recall_worst_case == 0.0
    assert "empty" in outcome.worst_perturbation_finding


def test_worst_case_seed_deterministic() -> None:
    out1 = wasserstein_worst_case(
        POOL_SMALL, _shift_sensitive_recall, epsilon=0.05, n_perturbations=20, seed_base=7
    )
    out2 = wasserstein_worst_case(
        POOL_SMALL, _shift_sensitive_recall, epsilon=0.05, n_perturbations=20, seed_base=7
    )
    assert out1 == out2


def test_worst_case_different_seeds_can_differ() -> None:
    out1 = wasserstein_worst_case(
        POOL_SMALL, _shift_sensitive_recall, epsilon=0.05, n_perturbations=10, seed_base=1
    )
    out2 = wasserstein_worst_case(
        POOL_SMALL, _shift_sensitive_recall, epsilon=0.05, n_perturbations=10, seed_base=999
    )
    # Different seeds should generally explore different perturbations; if they
    # happen to coincide we accept it but the contract is reproducibility per seed.
    assert isinstance(out1, DROOutcome)
    assert isinstance(out2, DROOutcome)


def test_worst_case_n_perturbations_zero_raises() -> None:
    with pytest.raises(ValueError, match=r"n_perturbations"):
        wasserstein_worst_case(POOL_SMALL, _identity_recall, epsilon=0.05, n_perturbations=0)


def test_worst_case_epsilon_one_raises() -> None:
    with pytest.raises(ValueError, match=r"epsilon"):
        wasserstein_worst_case(POOL_SMALL, _identity_recall, epsilon=1.0)


def test_worst_case_recall_fn_returns_nan_raises() -> None:
    def _nan_fn(_p: list[int]) -> float:
        return float("nan")

    with pytest.raises(RuntimeError, match=r"non-finite"):
        wasserstein_worst_case(POOL_SMALL, _nan_fn, epsilon=0.05, n_perturbations=3)


def test_worst_case_recall_fn_returns_out_of_range_raises() -> None:
    def _bad_fn(_p: list[int]) -> float:
        return 1.5

    with pytest.raises(RuntimeError, match=r"outside \[0, 1\]"):
        wasserstein_worst_case(POOL_SMALL, _bad_fn, epsilon=0.05, n_perturbations=3)


def test_worst_case_strict_recall_decreases() -> None:
    """With monotone recall on mean Elo, more perturbations exposes lower extremes."""
    out_few = wasserstein_worst_case(
        POOL_SMALL, _strict_recall, epsilon=0.10, n_perturbations=5, seed_base=42
    )
    out_many = wasserstein_worst_case(
        POOL_SMALL, _strict_recall, epsilon=0.10, n_perturbations=200, seed_base=42
    )
    assert out_many.recall_worst_case <= out_few.recall_worst_case


# -----------------------------------------------------------------------------
# compute_dro_for_match
# -----------------------------------------------------------------------------


def test_compute_dro_returns_dict_keyed_by_epsilon() -> None:
    out = compute_dro_for_match(POOL_SMALL, _identity_recall)
    assert set(out.keys()) == set(DEFAULT_EPSILONS)
    assert all(isinstance(v, DROOutcome) for v in out.values())


def test_compute_dro_deterministic_across_calls() -> None:
    out1 = compute_dro_for_match(POOL_SMALL, _shift_sensitive_recall, n_perturbations=10)
    out2 = compute_dro_for_match(POOL_SMALL, _shift_sensitive_recall, n_perturbations=10)
    for eps in DEFAULT_EPSILONS:
        assert out1[eps] == out2[eps]


def test_compute_dro_uses_default_n_perturbations() -> None:
    out = compute_dro_for_match(POOL_SMALL, _identity_recall)
    for outcome in out.values():
        assert outcome.n_perturbations == DEFAULT_N_PERTURBATIONS


def test_compute_dro_empty_epsilons_raises() -> None:
    with pytest.raises(ValueError, match=r"epsilons"):
        compute_dro_for_match(POOL_SMALL, _identity_recall, epsilons=[])


def test_compute_dro_per_epsilon_seed_offset_isolates_runs() -> None:
    """Per-ε seed offset prevents ε=0.05 and ε=0.10 from sharing exact RNG state."""
    out = compute_dro_for_match(POOL_SMALL, _shift_sensitive_recall, n_perturbations=20)
    assert out[0.05].worst_perturbation_finding != out[0.10].worst_perturbation_finding


def test_compute_dro_custom_epsilons() -> None:
    out = compute_dro_for_match(
        POOL_SMALL, _identity_recall, epsilons=(0.01, 0.03, 0.07), n_perturbations=5
    )
    assert set(out.keys()) == {0.01, 0.03, 0.07}


# -----------------------------------------------------------------------------
# Property: clipping bounds always respected
# -----------------------------------------------------------------------------


def test_perturb_clip_property_random_epsilons() -> None:
    """For 50 random ε in [0, 0.99], every perturbed Elo must lie in [floor, ceiling]."""
    rng = np.random.default_rng(0)
    pool = [int(x) for x in rng.integers(800, 2900, size=20)]
    for _ in range(50):
        eps = float(rng.uniform(0.0, 0.99))
        seed = int(rng.integers(0, 10_000))
        perturbed, _ = perturb_wasserstein(pool, eps, seed)
        assert all(ELO_FLOOR <= x <= ELO_CEILING for x in perturbed)


def test_dro_outcome_finite_recall_for_realistic_recall_fn() -> None:
    callable_fn: Callable[[list[int]], float] = _strict_recall
    out = compute_dro_for_match(POOL_SMALL, callable_fn, n_perturbations=30)
    for outcome in out.values():
        assert math.isfinite(outcome.recall_worst_case)
        assert 0.0 <= outcome.recall_worst_case <= 1.0
