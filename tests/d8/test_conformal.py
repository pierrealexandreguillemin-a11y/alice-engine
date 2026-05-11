"""Tests scripts/d8/conformal (Vovk 2024, Angelopoulos 2023).

Document ID: ALICE-D8-TEST-CONFORMAL
Version: 1.0.0
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from scripts.d8.conformal import (
    ConformalCalibration,
    conformal_set_size_mean,
    coverage_per_group,
    coverage_rate,
    split_calibrate,
)


def test_split_calibrate_returns_calibration_object() -> None:
    """Split conformal returns ConformalCalibration with correct fields."""
    rng = np.random.default_rng(42)
    n = 100
    y_obs = rng.uniform(0, 1, n)
    y_pred = rng.uniform(0, 1, n)
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    assert isinstance(cal, ConformalCalibration)
    assert cal.n_calibration == n
    assert cal.alpha == 0.10
    assert cal.quantile_threshold > 0.0


def test_split_calibrate_alpha_zero_raises() -> None:
    with pytest.raises(ValueError, match=r"alpha"):
        split_calibrate(np.array([0.5]), np.array([0.4]), alpha=0.0)


def test_split_calibrate_alpha_one_raises() -> None:
    with pytest.raises(ValueError, match=r"alpha"):
        split_calibrate(np.array([0.5]), np.array([0.4]), alpha=1.0)


def test_split_calibrate_alpha_negative_raises() -> None:
    with pytest.raises(ValueError, match=r"alpha"):
        split_calibrate(np.array([0.5]), np.array([0.4]), alpha=-0.1)


def test_split_calibrate_alpha_above_one_raises() -> None:
    with pytest.raises(ValueError, match=r"alpha"):
        split_calibrate(np.array([0.5]), np.array([0.4]), alpha=1.1)


def test_split_calibrate_empty_raises() -> None:
    with pytest.raises(ValueError, match=r"empty"):
        split_calibrate(np.array([]), np.array([]), alpha=0.10)


def test_split_calibrate_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"length"):
        split_calibrate(np.array([0.5, 0.6]), np.array([0.4]), alpha=0.10)


def test_coverage_rate_marginal_at_least_target() -> None:
    """Marginal coverage on calibration data ~ 1-alpha (Vovk 2024 Theorem 2.1)."""
    rng = np.random.default_rng(7)
    n = 1000
    y_obs = rng.uniform(0, 1, n)
    y_pred = y_obs + rng.normal(0, 0.05, n)
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    # Test on independent set
    y_obs_test = rng.uniform(0, 1, n)
    y_pred_test = y_obs_test + rng.normal(0, 0.05, n)
    cov = coverage_rate(y_obs_test, y_pred_test, cal)
    # Coverage should be approximately 1-alpha = 0.90 (split-conformal, ±0.05 sample)
    assert 0.80 < cov < 1.0


def test_coverage_per_group_returns_dict() -> None:
    rng = np.random.default_rng(1)
    n = 100
    y_obs = rng.uniform(0, 1, n)
    y_pred = y_obs + rng.normal(0, 0.05, n)
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    groups = ["A"] * 50 + ["B"] * 50
    result = coverage_per_group(y_obs, y_pred, groups, cal)
    assert "A" in result
    assert "B" in result


def test_coverage_per_group_empty_returns_dict() -> None:
    rng = np.random.default_rng(2)
    y_obs = rng.uniform(0, 1, 50)
    y_pred = rng.uniform(0, 1, 50)
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    result = coverage_per_group(np.array([]), np.array([]), [], cal)
    assert result == {}


def test_conformal_set_size_mean_within_bounds() -> None:
    """Mean conformal set size should be in (0, 1] (Angelopoulos 2023 §4.2)."""
    rng = np.random.default_rng(3)
    n = 200
    y_obs = rng.uniform(0, 1, n)
    y_pred = y_obs + rng.normal(0, 0.05, n)
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    size = conformal_set_size_mean(y_pred, cal, grid_resolution=0.01)
    assert 0.0 < size < 1.0


def test_calibration_dataclass_frozen() -> None:
    cal = ConformalCalibration(
        quantile_threshold=0.1,
        nonconf_scores=np.array([0.05, 0.10, 0.15]),
        n_calibration=3,
        alpha=0.10,
    )
    assert dataclasses.is_dataclass(cal)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cal.alpha = 0.20  # type: ignore[misc]


def test_quantile_rank_formula() -> None:
    """Quantile rank = ceil((1-alpha)*(n+1)) - 1 (Vovk 2024 §2.3).

    n=10, alpha=0.1 -> ceil(0.9*11) - 1 = ceil(9.9) - 1 = 10 - 1 = 9 (last sorted score).
    """
    y_obs = np.arange(10, dtype=float) / 10.0
    y_pred = np.zeros(10)
    # Nonconf scores = |y_obs - y_pred| = 0, 0.1, 0.2, ..., 0.9 (sorted)
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    assert math.isclose(cal.quantile_threshold, 0.9, abs_tol=1e-9)


def test_split_calibrate_nan_raises() -> None:
    with pytest.raises(ValueError, match="NaN or inf"):
        split_calibrate(np.array([0.5, float("nan")]), np.array([0.4, 0.5]), alpha=0.10)


def test_split_calibrate_inf_raises() -> None:
    with pytest.raises(ValueError, match="NaN or inf"):
        split_calibrate(np.array([0.5, float("inf")]), np.array([0.4, 0.5]), alpha=0.10)


def test_split_calibrate_nan_in_predicted_raises() -> None:
    with pytest.raises(ValueError, match="NaN or inf"):
        split_calibrate(np.array([0.5, 0.6]), np.array([0.4, float("nan")]), alpha=0.10)


def test_single_point_calibration_works() -> None:
    """Single calibration point: quantile = single nonconf score."""
    y_obs = np.array([0.5])
    y_pred = np.array([0.3])
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    # Single point, rank = ceil(0.9*2) - 1 = 2 - 1 = 1, clipped to 0
    assert cal.quantile_threshold >= 0.0
    assert cal.n_calibration == 1


def test_perfect_prediction_small_threshold() -> None:
    """Perfect predictions -> nonconf scores all 0 -> threshold ~ 0."""
    y_obs = np.linspace(0, 1, 100)
    y_pred = y_obs.copy()
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    assert math.isclose(cal.quantile_threshold, 0.0, abs_tol=1e-9)


def test_all_wrong_large_threshold() -> None:
    """Maximum miscalibration -> threshold = max nonconf score."""
    y_obs = np.zeros(100)
    y_pred = np.ones(100)
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    assert math.isclose(cal.quantile_threshold, 1.0, abs_tol=1e-9)


def test_alpha_005_targets_95_percent_coverage() -> None:
    """alpha=0.05 -> 95% target coverage (Vovk 2024 §2.3)."""
    rng = np.random.default_rng(11)
    n = 2000
    y_obs = rng.uniform(0, 1, n)
    y_pred = y_obs + rng.normal(0, 0.03, n)
    cal = split_calibrate(y_obs, y_pred, alpha=0.05)
    # Independent test set
    y_obs_test = rng.uniform(0, 1, n)
    y_pred_test = y_obs_test + rng.normal(0, 0.03, n)
    cov = coverage_rate(y_obs_test, y_pred_test, cal)
    assert cov >= 0.85
    assert cal.alpha == 0.05


def test_alpha_05_targets_50_percent_coverage() -> None:
    """alpha=0.5 -> 50% target coverage."""
    rng = np.random.default_rng(13)
    n = 2000
    y_obs = rng.uniform(0, 1, n)
    y_pred = y_obs + rng.normal(0, 0.05, n)
    cal = split_calibrate(y_obs, y_pred, alpha=0.5)
    # Independent test set
    y_obs_test = rng.uniform(0, 1, n)
    y_pred_test = y_obs_test + rng.normal(0, 0.05, n)
    cov = coverage_rate(y_obs_test, y_pred_test, cal)
    # 50% coverage with reasonable tolerance for finite sample
    assert 0.40 < cov < 0.60


def test_large_dataset_coverage() -> None:
    """Large N=1000+: coverage should converge to 1-alpha."""
    rng = np.random.default_rng(17)
    n = 5000
    y_obs = rng.uniform(0, 1, n)
    y_pred = y_obs + rng.normal(0, 0.04, n)
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    y_obs_test = rng.uniform(0, 1, n)
    y_pred_test = y_obs_test + rng.normal(0, 0.04, n)
    cov = coverage_rate(y_obs_test, y_pred_test, cal)
    # Tighter bound at large N
    assert 0.87 < cov < 0.93


def test_coverage_rate_length_mismatch_raises() -> None:
    cal = ConformalCalibration(
        quantile_threshold=0.1,
        nonconf_scores=np.array([0.05, 0.10]),
        n_calibration=2,
        alpha=0.10,
    )
    with pytest.raises(ValueError, match=r"length"):
        coverage_rate(np.array([0.5, 0.6]), np.array([0.4]), cal)


def test_coverage_per_group_length_mismatch_raises() -> None:
    cal = ConformalCalibration(
        quantile_threshold=0.1,
        nonconf_scores=np.array([0.05, 0.10]),
        n_calibration=2,
        alpha=0.10,
    )
    with pytest.raises(ValueError, match=r"length"):
        coverage_per_group(np.array([0.5, 0.6]), np.array([0.4, 0.5]), ["A"], cal)


def test_coverage_per_group_unbalanced() -> None:
    rng = np.random.default_rng(19)
    n = 100
    y_obs = rng.uniform(0, 1, n)
    y_pred = y_obs + rng.normal(0, 0.05, n)
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    groups = ["A"] * 90 + ["B"] * 10
    result = coverage_per_group(y_obs, y_pred, groups, cal)
    assert "A" in result
    assert "B" in result
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_conformal_set_size_perfect_predictor_small() -> None:
    """Perfect predictor -> threshold = 0 -> set size very small."""
    y_obs = np.linspace(0, 1, 200)
    y_pred = y_obs.copy()
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    size = conformal_set_size_mean(y_pred, cal, grid_resolution=0.01)
    # Threshold 0 -> single point only
    assert size <= 0.05


def test_conformal_set_size_grid_resolution_validation() -> None:
    """ISO 27034 input validation : grid_resolution must be in (0, 1]."""
    cal = ConformalCalibration(
        quantile_threshold=0.1,
        nonconf_scores=np.array([0.05, 0.10]),
        n_calibration=2,
        alpha=0.10,
    )
    with pytest.raises(ValueError, match=r"grid_resolution"):
        conformal_set_size_mean(np.array([0.5]), cal, grid_resolution=0.0)


def test_nonconf_scores_sorted() -> None:
    """ConformalCalibration.nonconf_scores should be sorted ascending."""
    rng = np.random.default_rng(23)
    n = 50
    y_obs = rng.uniform(0, 1, n)
    y_pred = rng.uniform(0, 1, n)
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    sorted_scores = np.sort(cal.nonconf_scores)
    assert np.array_equal(cal.nonconf_scores, sorted_scores)


# D-2026-05-11 support_max fix tests (E[score] match sum ∈ [0, K=team_size])


def test_conformal_set_size_support_max_k_boards() -> None:
    """E[score] match sum ∈ [0, K=8] : set_size doit refléter K, pas saturer à 1.

    D-2026-05-11 : avant le fix, le clip [0, 1.0] saturait set_size_mean=1.0
    pour q_hat=4.4 sur range [0, 8]. Avec support_max=8.0, set_size_mean reflète
    réellement l'efficiency Angelopoulos 2023 §4.2.
    """
    rng = np.random.default_rng(42)
    n = 100
    # E[score] simulated on [0, 8] : 8 boards × per-board E[score] ∈ [0, 1]
    y_obs = rng.uniform(0, 8, n)
    y_pred = y_obs + rng.normal(0, 1.0, n)
    cal = split_calibrate(y_obs, y_pred, alpha=0.10)
    # Without support_max (default 1.0), saturates at 1.0
    size_clipped = conformal_set_size_mean(y_pred, cal, support_max=1.0)
    # With support_max=8.0, reflects real interval width
    size_real = conformal_set_size_mean(y_pred, cal, support_max=8.0)
    assert size_real > size_clipped, (
        f"support_max fix should expand set_size : clipped={size_clipped:.3f} "
        f"vs real={size_real:.3f} (q_hat={cal.quantile_threshold:.3f})"
    )
    assert 0.0 < size_real <= 8.0


def test_conformal_set_size_support_max_zero_raises() -> None:
    """ISO 27034 input validation : support_max must be > 0."""
    cal = ConformalCalibration(
        quantile_threshold=0.5,
        nonconf_scores=np.array([0.1, 0.5]),
        n_calibration=2,
        alpha=0.10,
    )
    with pytest.raises(ValueError, match=r"support_max"):
        conformal_set_size_mean(np.array([4.0]), cal, support_max=0.0)


def test_conformal_set_size_support_max_negative_raises() -> None:
    cal = ConformalCalibration(
        quantile_threshold=0.5,
        nonconf_scores=np.array([0.1, 0.5]),
        n_calibration=2,
        alpha=0.10,
    )
    with pytest.raises(ValueError, match=r"support_max"):
        conformal_set_size_mean(np.array([4.0]), cal, support_max=-1.0)


def test_conformal_set_size_clipped_at_support_max() -> None:
    """Set bounds clipped to [0, support_max] : never exceeds support range."""
    cal = ConformalCalibration(
        quantile_threshold=10.0,  # huge q_hat = full coverage
        nonconf_scores=np.array([5.0, 10.0]),
        n_calibration=2,
        alpha=0.10,
    )
    # Even with q=10, y_pred=4, support_max=8 → set = [0, 8] = 8.0
    size = conformal_set_size_mean(np.array([4.0]), cal, support_max=8.0)
    assert size == pytest.approx(8.0)
