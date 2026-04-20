"""Tests T14 Smoke robustness — Plan 3 V2 Phase 3 (ISO 24029)."""

from __future__ import annotations

import pytest

from scripts.backtest.robustness import (
    ROBUSTNESS_RECALL_DROP_GATE,
    RobustnessResult,
    compute_recall_drop,
    perturb_elos,
    robustness_smoke,
)


def test_perturb_elos_reproducible_same_seed():
    elos = [1500, 1800, 2000, 2200]
    out1 = perturb_elos(elos, noise_pct=0.05, seed=42)
    out2 = perturb_elos(elos, noise_pct=0.05, seed=42)
    assert out1 == out2


def test_perturb_elos_different_seeds_differ():
    elos = [1500, 1800, 2000, 2200]
    out1 = perturb_elos(elos, noise_pct=0.05, seed=42)
    out2 = perturb_elos(elos, noise_pct=0.05, seed=123)
    assert out1 != out2


def test_perturb_elos_clips_to_ffe_range():
    elos = [900, 2800]
    out = perturb_elos(elos, noise_pct=0.50, seed=42)
    assert all(800 <= e <= 2900 for e in out)


def test_perturb_elos_zero_noise_identity():
    elos = [1500, 1800, 2000]
    out = perturb_elos(elos, noise_pct=0.0, seed=42)
    assert out == elos


def test_perturb_elos_empty_returns_empty():
    assert perturb_elos([], noise_pct=0.1) == []


def test_perturb_elos_invalid_noise_raises():
    with pytest.raises(ValueError, match="noise_pct"):
        perturb_elos([1500], noise_pct=-0.1)
    with pytest.raises(ValueError, match="noise_pct"):
        perturb_elos([1500], noise_pct=1.0)


def test_compute_recall_drop_positive():
    assert compute_recall_drop(baseline_recall=0.9, perturbed_recall=0.8) == pytest.approx(0.1)


def test_compute_recall_drop_clamps_negative():
    # Perturbed improves -> drop = 0 (clamped)
    assert compute_recall_drop(baseline_recall=0.8, perturbed_recall=0.9) == 0.0


def test_robustness_smoke_passes_gate():
    res = robustness_smoke(baseline_recall=0.92, perturbed_recall=0.88, noise_pct=0.05)
    assert res.passes_gate(gate=ROBUSTNESS_RECALL_DROP_GATE) is True
    assert res.recall_drop == pytest.approx(0.04)


def test_robustness_smoke_fails_gate():
    res = robustness_smoke(baseline_recall=0.92, perturbed_recall=0.75, noise_pct=0.05)
    assert res.passes_gate() is False


def test_robustness_result_frozen():
    res = robustness_smoke(baseline_recall=0.9, perturbed_recall=0.85, noise_pct=0.05)
    assert isinstance(res, RobustnessResult)
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        res.recall_drop = 0.99  # type: ignore[misc]


def test_perturb_elos_mean_preserved_approx():
    """Gaussian noise centered 0 -> mean approx preserved sur large sample."""
    import numpy as np

    elos = [1800] * 200
    out = perturb_elos(elos, noise_pct=0.05, seed=42)
    # Mean should stay close (within 3% of original)
    assert abs(np.mean(out) - 1800) / 1800 < 0.03
