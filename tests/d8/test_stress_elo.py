"""Tests scripts/d8/stress_elo (Goodfellow 2015, Madry 2018, ISO 24029).

Document ID: ALICE-D8-TEST-STRESS-ELO
Version: 1.0.0
"""

from __future__ import annotations

import pytest

from scripts.d8.stress_elo import (
    NOISE_LEVELS,
    ElostressOutcome,
    compute_stress_elo_for_match,
    run_multinoise,
)

# ---- Constants & dataclass ----


def test_noise_levels_default_5() -> None:
    """Default noise levels = [0.01, 0.03, 0.05, 0.07, 0.10] (ISO 24029 §6.5)."""
    assert NOISE_LEVELS == [0.01, 0.03, 0.05, 0.07, 0.10]


def test_outcome_is_frozen_dataclass() -> None:
    out = ElostressOutcome(
        noise_pct=0.05,
        baseline_recall=0.80,
        perturbed_recall_mean=0.78,
        recall_drop=0.02,
    )
    with pytest.raises(AttributeError):
        out.noise_pct = 0.10  # type: ignore[misc]


# ---- compute_stress_elo_for_match ----


def test_stress_elo_match_returns_dict_keyed_by_noise() -> None:
    pool = [1500, 1600, 1700, 1800]

    def backtest_fn(_perturbed: list[int]) -> float:
        return 0.75

    result = compute_stress_elo_for_match(pool, backtest_fn)
    assert set(result.keys()) == set(NOISE_LEVELS)


def test_stress_elo_match_determinism_same_seed() -> None:
    pool = [1500, 1600, 1700, 1800, 1900]
    calls_a: list[list[int]] = []
    calls_b: list[list[int]] = []

    def fn_a(p: list[int]) -> float:
        calls_a.append(list(p))
        return 0.5

    def fn_b(p: list[int]) -> float:
        calls_b.append(list(p))
        return 0.5

    compute_stress_elo_for_match(pool, fn_a, seed=123)
    compute_stress_elo_for_match(pool, fn_b, seed=123)
    assert calls_a == calls_b


def test_stress_elo_match_custom_noise_levels() -> None:
    pool = [1500, 1600]

    def backtest_fn(_p: list[int]) -> float:
        return 0.6

    result = compute_stress_elo_for_match(pool, backtest_fn, noise_levels=[0.02, 0.08])
    assert set(result.keys()) == {0.02, 0.08}


def test_stress_elo_match_empty_pool_returns_dict_with_recalls() -> None:
    """Empty pool -> perturb_elos returns []; backtest_fn called with []."""

    def backtest_fn(p: list[int]) -> float:
        assert p == []
        return 0.0

    result = compute_stress_elo_for_match([], backtest_fn, noise_levels=[0.05])
    assert result == {0.05: 0.0}


def test_stress_elo_match_invalid_noise_negative_raises() -> None:
    pool = [1500, 1600]

    def fn(_p: list[int]) -> float:
        return 0.5

    with pytest.raises(ValueError, match=r"noise_pct must be in"):
        compute_stress_elo_for_match(pool, fn, noise_levels=[-0.01])


def test_stress_elo_match_invalid_noise_above_one_raises() -> None:
    pool = [1500, 1600]

    def fn(_p: list[int]) -> float:
        return 0.5

    with pytest.raises(ValueError, match=r"noise_pct must be in"):
        compute_stress_elo_for_match(pool, fn, noise_levels=[1.0])


def test_stress_elo_match_zero_noise_returns_unperturbed() -> None:
    """noise_pct=0 -> perturb_elos returns same input (sigma=0)."""
    pool = [1500, 1600, 1700]
    received: list[list[int]] = []

    def fn(p: list[int]) -> float:
        received.append(list(p))
        return 0.5

    compute_stress_elo_for_match(pool, fn, noise_levels=[0.0])
    assert received[0] == pool


# ---- run_multinoise ----


def test_run_multinoise_returns_outcomes_sorted_by_noise() -> None:
    baseline = [0.80, 0.82, 0.78]
    perturbed = {
        0.10: [0.65, 0.66, 0.64],
        0.01: [0.79, 0.81, 0.77],
        0.05: [0.72, 0.73, 0.71],
    }
    out = run_multinoise(baseline, perturbed)
    noises = [o.noise_pct for o in out]
    assert noises == sorted(noises)


def test_run_multinoise_baseline_recall_uniform() -> None:
    """All outcomes share same baseline_recall (mean of baseline list)."""
    baseline = [0.80, 0.80, 0.80]
    perturbed = {0.05: [0.70], 0.10: [0.60]}
    out = run_multinoise(baseline, perturbed)
    assert all(o.baseline_recall == pytest.approx(0.80, abs=1e-9) for o in out)


def test_run_multinoise_recall_drop_clamped_to_zero() -> None:
    """If perturbed > baseline, recall_drop = 0 (clamp)."""
    baseline = [0.50]
    perturbed = {0.05: [0.60]}
    out = run_multinoise(baseline, perturbed)
    assert out[0].recall_drop == 0.0


def test_run_multinoise_recall_drop_positive_when_degraded() -> None:
    baseline = [0.80, 0.80]
    perturbed = {0.10: [0.60, 0.60]}
    out = run_multinoise(baseline, perturbed)
    assert out[0].recall_drop == pytest.approx(0.20, abs=1e-9)


def test_run_multinoise_perturbed_recall_is_mean() -> None:
    baseline = [0.50]
    perturbed = {0.05: [0.40, 0.50, 0.60]}
    out = run_multinoise(baseline, perturbed)
    assert out[0].perturbed_recall_mean == pytest.approx(0.50, abs=1e-9)


def test_run_multinoise_empty_inputs_return_empty_list() -> None:
    out = run_multinoise([], {})
    assert out == []


def test_run_multinoise_single_noise_returns_single_outcome() -> None:
    baseline = [0.70]
    perturbed = {0.03: [0.65]}
    out = run_multinoise(baseline, perturbed)
    assert len(out) == 1
    assert out[0].noise_pct == 0.03
