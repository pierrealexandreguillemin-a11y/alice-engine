"""Tests scripts/d8/stress_roster (Tran 2022, ISO 24029 §6.5).

Document ID: ALICE-D8-TEST-STRESS-ROSTER
Version: 1.0.0
"""

from __future__ import annotations

import pytest

from scripts.d8.stress_roster import (
    TURNOVER_RATES,
    RosterStressOutcome,
    aggregate_roster_outcomes,
    compute_stress_roster_for_match,
    drop_random_players,
)

# ---- Constants & dataclass ----


def test_turnover_rates_default() -> None:
    assert TURNOVER_RATES == [0.05, 0.10, 0.20]


def test_outcome_is_frozen_dataclass() -> None:
    out = RosterStressOutcome(
        turnover_pct=0.10,
        baseline_recall=0.80,
        perturbed_recall_mean=0.78,
        recall_drop=0.02,
    )
    with pytest.raises(AttributeError):
        out.turnover_pct = 0.20  # type: ignore[misc]


# ---- drop_random_players ----


def test_drop_random_players_determinism() -> None:
    pool = list(range(20))
    a = drop_random_players(pool, 0.20, seed=7)
    b = drop_random_players(pool, 0.20, seed=7)
    assert a == b


def test_drop_random_players_different_seed_different_output() -> None:
    pool = list(range(50))
    a = drop_random_players(pool, 0.20, seed=1)
    b = drop_random_players(pool, 0.20, seed=2)
    assert a != b


def test_drop_random_players_zero_drop_returns_full_pool() -> None:
    pool = [1, 2, 3, 4, 5]
    out = drop_random_players(pool, 0.0, seed=42)
    assert sorted(out) == sorted(pool)


def test_drop_random_players_negative_raises() -> None:
    with pytest.raises(ValueError, match=r"turnover_pct"):
        drop_random_players([1, 2, 3], -0.1, seed=42)


def test_drop_random_players_one_or_above_raises() -> None:
    with pytest.raises(ValueError, match=r"turnover_pct"):
        drop_random_players([1, 2, 3], 1.0, seed=42)


def test_drop_random_players_empty_pool_returns_empty() -> None:
    assert drop_random_players([], 0.10, seed=42) == []


def test_drop_random_players_drop_count_floor() -> None:
    """20% of 10 = 2 players dropped -> remaining 8."""
    pool = list(range(10))
    out = drop_random_players(pool, 0.20, seed=42)
    assert len(out) == 8


# ---- compute_stress_roster_for_match ----


def test_compute_stress_roster_returns_dict_keyed_by_turnover() -> None:
    pool = list(range(20))

    def fn(_p: list[int]) -> float:
        return 0.6

    result = compute_stress_roster_for_match(pool, fn, min_pool_size=4)
    assert set(result.keys()) == set(TURNOVER_RATES)


def test_compute_stress_roster_skips_when_below_min_pool_size() -> None:
    """Pool too small after drop -> None per turnover."""
    pool = [1, 2, 3]  # n=3, min=4 -> always None

    def fn(_p: list[int]) -> float:
        return 0.5

    result = compute_stress_roster_for_match(pool, fn, min_pool_size=4)
    assert all(v is None for v in result.values())


# ---- aggregate_roster_outcomes ----


def test_aggregate_roster_outcomes_sorted_by_turnover() -> None:
    baseline = [0.80, 0.78]
    perturbed = {
        0.20: [0.60, 0.62],
        0.05: [0.78, 0.76],
        0.10: [0.70, 0.68],
    }
    out = aggregate_roster_outcomes(baseline, perturbed)
    turnovers = [o.turnover_pct for o in out]
    assert turnovers == sorted(turnovers)


def test_aggregate_roster_outcomes_recall_drop_clamped() -> None:
    """If perturbed > baseline, drop = 0."""
    baseline = [0.50]
    perturbed = {0.10: [0.60]}
    out = aggregate_roster_outcomes(baseline, perturbed)
    assert out[0].recall_drop == 0.0
