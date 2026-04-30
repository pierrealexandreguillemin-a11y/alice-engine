"""Tests scripts/d8/calibration (Pleiss 2017, Hebert-Johnson 2018).

Document ID: ALICE-D8-TEST-CALIBRATION
Version: 1.0.0
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.d8.calibration import (
    compute_ece_per_group,
    compute_multicalibration_alpha,
    expected_calibration_error,
)


def test_ece_perfect_calibration_returns_zero() -> None:
    probs = np.array([0.1, 0.5, 0.9])
    labels = np.array([0, 1, 1])
    ece = expected_calibration_error(probs, labels, n_bins=2)
    assert ece >= 0.0


def test_ece_total_miscalibration_high() -> None:
    """All probas 0.9 but labels 0 -> ECE close to 0.9."""
    probs = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    labels = np.array([0, 0, 0, 0, 0])
    ece = expected_calibration_error(probs, labels, n_bins=2)
    assert ece > 0.5


def test_ece_empty_input_zero() -> None:
    ece = expected_calibration_error(np.array([]), np.array([]), n_bins=10)
    assert ece == 0.0


def test_ece_n_bins_validation() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5]), np.array([1]), n_bins=0)


def test_ece_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5, 0.6]), np.array([1]), n_bins=2)


def test_ece_per_group_returns_dict() -> None:
    probs = np.array([0.5, 0.7, 0.3, 0.9])
    labels = np.array([1, 1, 0, 1])
    groups = ["M", "M", "F", "F"]
    result = compute_ece_per_group(probs, labels, groups, n_bins=2)
    assert "M" in result
    assert "F" in result


def test_ece_per_group_values_finite() -> None:
    probs = np.array([0.5, 0.7])
    labels = np.array([1, 1])
    groups = ["M", "M"]
    result = compute_ece_per_group(probs, labels, groups, n_bins=2)
    assert math.isfinite(result["M"])


def test_ece_per_group_empty() -> None:
    result = compute_ece_per_group(np.array([]), np.array([]), [], n_bins=10)
    assert result == {}


def test_multicalibration_alpha_returns_float() -> None:
    probs = np.array([0.5, 0.6, 0.7, 0.8])
    labels = np.array([1, 0, 1, 1])
    groups = {"all": np.array([True, True, True, True])}
    alpha = compute_multicalibration_alpha(probs, labels, groups, n_bins=2)
    assert math.isfinite(alpha)


def test_multicalibration_alpha_perfect_zero() -> None:
    """Perfect calibration -> alpha close to 0."""
    n = 1000
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < probs).astype(int)
    groups = {"all": np.ones(n, dtype=bool)}
    alpha = compute_multicalibration_alpha(probs, labels, groups, n_bins=10)
    assert alpha < 0.10


def test_multicalibration_alpha_subgroup_violation() -> None:
    """Subgroup miscalibration -> high alpha."""
    n = 200
    probs = np.full(n, 0.9)
    labels = np.zeros(n, dtype=int)
    groups = {"sub": np.ones(n, dtype=bool)}
    alpha = compute_multicalibration_alpha(probs, labels, groups, n_bins=2)
    assert alpha > 0.5


def test_ece_n_bins_too_high_clips() -> None:
    probs = np.array([0.5])
    labels = np.array([1])
    ece = expected_calibration_error(probs, labels, n_bins=100)
    assert math.isfinite(ece)


def test_ece_constant_proba_zero() -> None:
    probs = np.full(10, 0.5)
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert math.isfinite(ece)


def test_ece_per_group_unbalanced_groups() -> None:
    probs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    labels = np.array([1, 1, 1, 1, 1])
    groups = ["M", "M", "M", "M", "F"]
    result = compute_ece_per_group(probs, labels, groups, n_bins=2)
    assert "M" in result
    assert "F" in result


def test_multicalibration_alpha_multiple_groups() -> None:
    n = 100
    probs = np.linspace(0, 1, n)
    labels = (probs > 0.5).astype(int)
    groups = {
        "g1": np.array([True] * 50 + [False] * 50),
        "g2": np.array([False] * 50 + [True] * 50),
    }
    alpha = compute_multicalibration_alpha(probs, labels, groups, n_bins=5)
    assert math.isfinite(alpha)


def test_multicalibration_empty_groups_zero() -> None:
    n = 100
    probs = np.full(n, 0.5)
    labels = np.zeros(n, dtype=int)
    alpha = compute_multicalibration_alpha(probs, labels, groups={}, n_bins=2)
    assert alpha == 0.0


def test_ece_threshold_strict() -> None:
    """ISO Pleiss 2017 §4 : ECE 0.05 should be borderline."""
    rng = np.random.default_rng(1)
    n = 500
    probs = rng.uniform(0, 1, n)
    # Inject 5% miscalibration : labels = (rng < probs - 0.05)
    labels = (rng.uniform(0, 1, n) < (probs - 0.05).clip(0, 1)).astype(int)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert 0.01 < ece < 0.15


def test_ece_per_group_returns_n_groups_keys() -> None:
    probs = np.array([0.5, 0.7])
    labels = np.array([1, 1])
    groups = ["M", "F"]
    result = compute_ece_per_group(probs, labels, groups, n_bins=2)
    assert len(result) == 2


def test_multicalibration_alpha_returns_max() -> None:
    """Alpha = max ECE across subgroups (Hebert-Johnson 2018 §3)."""
    n = 100
    probs = np.full(n, 0.5)
    labels = np.zeros(n, dtype=int)
    g1_calibrated = np.ones(50, dtype=bool)
    g2_miscalibrated = np.zeros(50, dtype=bool)
    groups = {
        "g1": np.concatenate([g1_calibrated, np.zeros(50, dtype=bool)]),
        "g2": np.concatenate([np.zeros(50, dtype=bool), g2_miscalibrated]),
    }
    alpha = compute_multicalibration_alpha(probs, labels, groups, n_bins=2)
    assert alpha >= 0.0


def test_ece_per_group_n_bins_validation() -> None:
    with pytest.raises(ValueError):
        compute_ece_per_group(np.array([0.5]), np.array([1]), ["M"], n_bins=0)


def test_ece_nan_probs_raises() -> None:
    probs = np.array([0.5, float("nan"), 0.7])
    labels = np.array([1, 1, 1])
    with pytest.raises(ValueError, match="NaN or inf"):
        expected_calibration_error(probs, labels, n_bins=10)


def test_ece_inf_probs_raises() -> None:
    probs = np.array([0.5, float("inf"), 0.7])
    labels = np.array([1, 1, 1])
    with pytest.raises(ValueError, match="NaN or inf"):
        expected_calibration_error(probs, labels, n_bins=10)


def test_ece_probs_out_of_range_high_raises() -> None:
    probs = np.array([0.5, 1.5, 0.7])
    labels = np.array([1, 1, 1])
    with pytest.raises(ValueError, match=r"out of \[0, 1\]"):
        expected_calibration_error(probs, labels, n_bins=10)


def test_ece_probs_out_of_range_low_raises() -> None:
    probs = np.array([0.5, -0.1, 0.7])
    labels = np.array([1, 1, 1])
    with pytest.raises(ValueError, match=r"out of \[0, 1\]"):
        expected_calibration_error(probs, labels, n_bins=10)
