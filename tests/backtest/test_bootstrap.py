"""Tests T8 Bootstrap CI (BCa) — Plan 3 V2 Phase 3."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.backtest.bootstrap import bootstrap_ci


def test_bootstrap_ci_mean_is_point_estimate():
    """Point estimate == mean of original sample."""
    values = [0.8, 0.85, 0.9, 0.75, 0.95, 0.7, 0.88, 0.92, 0.83, 0.87]
    ci = bootstrap_ci(values, confidence=0.95, n_resamples=200, seed=42)
    assert ci.point == pytest.approx(np.mean(values))
    assert ci.n_samples == 10
    assert ci.method == "BCa"


def test_bootstrap_ci_lower_leq_point_leq_upper():
    """Invariant : lower ≤ point ≤ upper."""
    values = np.random.default_rng(0).normal(0.85, 0.05, size=50).tolist()
    ci = bootstrap_ci(values, n_resamples=500, seed=42)
    assert ci.lower <= ci.point <= ci.upper


def test_bootstrap_ci_reproducible_same_seed():
    """Same seed → identical CI (ISO 29119 determinism)."""
    values = [0.7, 0.8, 0.9, 0.85, 0.75] * 5
    ci1 = bootstrap_ci(values, n_resamples=300, seed=42)
    ci2 = bootstrap_ci(values, n_resamples=300, seed=42)
    assert ci1.lower == ci2.lower
    assert ci1.upper == ci2.upper


def test_bootstrap_ci_passes_gate_ge_direction():
    """Gate ge (≥ threshold) : PASS si ci.lower ≥ threshold."""
    # Sample centered at 0.92 → CI lower should be ≥ 0.90
    values = [0.92] * 30
    ci = bootstrap_ci(values, n_resamples=200, seed=42)
    assert ci.passes_gate(0.90, direction="ge") is True
    assert ci.passes_gate(0.95, direction="ge") is False


def test_bootstrap_ci_passes_gate_le_direction():
    """Gate le (≤ threshold) : PASS si ci.upper ≤ threshold."""
    values = [0.10] * 30
    ci = bootstrap_ci(values, n_resamples=200, seed=42)
    assert ci.passes_gate(0.20, direction="le") is True
    assert ci.passes_gate(0.05, direction="le") is False


def test_bootstrap_ci_empty_raises():
    """Empty values → ValueError."""
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])


def test_bootstrap_ci_single_sample_degenerate():
    """n=1 → CI dégénéré (lower=point=upper)."""
    ci = bootstrap_ci([0.85])
    assert ci.lower == ci.point == ci.upper == 0.85
    assert ci.n_resamples == 0


def test_bootstrap_ci_invalid_confidence_raises():
    """Confidence hors (0,1) → ValueError."""
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([0.5, 0.6], confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([0.5, 0.6], confidence=0.0)


def test_bootstrap_ci_dataclass_is_frozen():
    """BootstrapCI est frozen (ISO 29119 immutability)."""
    ci = bootstrap_ci([0.8, 0.9])
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        ci.lower = 0.5  # type: ignore[misc]


def test_bootstrap_ci_wide_variance_wider_ci():
    """Variance haute → CI plus large."""
    low_var = [0.85] * 20 + [0.87, 0.83]
    high_var = [0.5, 0.9, 0.6, 0.95, 0.55, 0.85, 0.7, 0.9] * 3
    ci_low = bootstrap_ci(low_var, n_resamples=300, seed=42)
    ci_high = bootstrap_ci(high_var, n_resamples=300, seed=42)
    width_low = ci_low.upper - ci_low.lower
    width_high = ci_high.upper - ci_high.lower
    assert width_high > width_low
