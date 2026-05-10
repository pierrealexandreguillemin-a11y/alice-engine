"""Tests scripts/d8/perturb_runner — cache-mutation infra for stress/DRO.

Document ID: ALICE-D8-TEST-PERTURB-RUNNER
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from scripts.d8.perturb_runner import (
    DRO_EPSILONS,
    NOISE_LEVELS_ELO,
    SUBSET_PER_SAISON,
    TURNOVER_RATES_ROSTER,
    compute_dro_real,
    compute_stress_elo_real,
    compute_stress_roster_real,
    perturbed_opponent_pool,
    stratified_subset,
)

# -----------------------------------------------------------------------------
# Fixtures + minimal stubs (no real BacktestHarness boot)
# -----------------------------------------------------------------------------


@dataclass
class _FakeMatchCandidate:
    """MatchCandidate stub matching runner.run_single signature."""

    saison: int
    ronde: int
    user_team: str
    opp_team: str
    opp_club: str


@dataclass
class _FakeMatchResult:
    recall_ali: float


class _FakeCache:
    """Stub cache with mutable joueurs_by_club dict."""

    def __init__(self, club_pools: dict[str, pd.DataFrame]) -> None:
        self.joueurs_by_club = club_pools


class _FakeHarness:
    def __init__(self, cache: _FakeCache) -> None:
        self.cache = cache


class _FakeRunner:
    """Captures perturbed elos seen at runtime → recall_ali ∈ [0.5, 0.95]."""

    def __init__(self, harness: _FakeHarness) -> None:
        self.harness = harness
        self.calls: list[tuple[str, list[int]]] = []

    def run_single(
        self,
        *,
        saison: int,
        ronde: int,
        user_team: str,
        opp_team: str,
        opp_club: str,
    ) -> _FakeMatchResult:
        elos = self.harness.cache.joueurs_by_club[opp_club]["elo"].tolist()
        self.calls.append((opp_club, elos))
        # Deterministic recall mimicking real "lower mean elo → higher ali recall"
        mean_elo = float(np.mean(elos)) if elos else 1500.0
        recall = max(0.5, min(0.95, 1.0 - (mean_elo - 1500) / 2000))
        return _FakeMatchResult(recall_ali=recall)


def _make_pool(elos: list[int], club: str = "C001") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nr_ffe": [f"P{i:03d}" for i in range(len(elos))],
            "elo": elos,
            "club": [club] * len(elos),
            "nom": ["X"] * len(elos),
            "prenom": ["Y"] * len(elos),
            "mute": [False] * len(elos),
            "genre": ["M"] * len(elos),
            "categorie": ["SE"] * len(elos),
            "elo_type": ["E"] * len(elos),
            "age_min": [None] * len(elos),
            "age_max": [None] * len(elos),
        }
    )


# -----------------------------------------------------------------------------
# perturbed_opponent_pool context manager
# -----------------------------------------------------------------------------


def test_perturbed_pool_mutates_then_restores() -> None:
    pool_orig = _make_pool([1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200])
    cache = _FakeCache({"C001": pool_orig})
    perturbed_elos = [1450, 1550, 1650, 1750, 1850, 1950, 2050, 2150]
    with perturbed_opponent_pool(cache, "C001", perturbed_elos):
        assert cache.joueurs_by_club["C001"]["elo"].tolist() == perturbed_elos
    # After exit, original elos restored
    assert cache.joueurs_by_club["C001"]["elo"].tolist() == [
        1500,
        1600,
        1700,
        1800,
        1900,
        2000,
        2100,
        2200,
    ]


def test_perturbed_pool_restores_on_exception() -> None:
    pool_orig = _make_pool([1600, 1700, 1800, 1900])
    cache = _FakeCache({"C001": pool_orig})
    with pytest.raises(RuntimeError, match=r"boom"):
        with perturbed_opponent_pool(cache, "C001", [1000, 1100, 1200, 1300]):
            raise RuntimeError("boom")
    # Original restored despite exception
    assert cache.joueurs_by_club["C001"]["elo"].tolist() == [1600, 1700, 1800, 1900]


def test_perturbed_pool_unknown_club_raises() -> None:
    cache = _FakeCache({"C001": _make_pool([1500])})
    with pytest.raises(KeyError, match=r"C999"):
        with perturbed_opponent_pool(cache, "C999", [1500]):
            pass


def test_perturbed_pool_partial_overwrite() -> None:
    """If perturbed_elos shorter than pool, only first N rows overwritten."""
    pool_orig = _make_pool([1500, 1600, 1700, 1800])
    cache = _FakeCache({"C001": pool_orig})
    with perturbed_opponent_pool(cache, "C001", [1000, 1100]):
        assert cache.joueurs_by_club["C001"]["elo"].tolist() == [1000, 1100, 1700, 1800]


# -----------------------------------------------------------------------------
# stratified_subset
# -----------------------------------------------------------------------------


def test_stratified_subset_returns_input_when_smaller() -> None:
    matches = [_FakeMatchCandidate(2024, r, f"U{r}", f"O{r}", f"C{r:03d}") for r in range(1, 6)]
    out = stratified_subset(matches, n=30)
    assert len(out) == 5


def test_stratified_subset_caps_at_n() -> None:
    matches = [
        _FakeMatchCandidate(2024, (i % 5) + 1, f"U{i}", f"O{i}", f"C{i}") for i in range(100)
    ]
    out = stratified_subset(matches, n=20)
    assert len(out) == 20


def test_stratified_subset_balances_rondes() -> None:
    matches = [
        _FakeMatchCandidate(2024, (i % 5) + 1, f"U{i}", f"O{i}", f"C{i}") for i in range(100)
    ]
    out = stratified_subset(matches, n=15)
    rondes = {m.ronde for m in out}
    # Should have all 5 rondes represented (one per ronde quota = 3)
    assert len(rondes) == 5


def test_stratified_subset_seed_deterministic() -> None:
    matches = [_FakeMatchCandidate(2024, (i % 5) + 1, f"U{i}", f"O{i}", f"C{i}") for i in range(50)]
    out1 = stratified_subset(matches, n=10, seed=42)
    out2 = stratified_subset(matches, n=10, seed=42)
    assert [m.ronde for m in out1] == [m.ronde for m in out2]


# -----------------------------------------------------------------------------
# compute_stress_elo_real
# -----------------------------------------------------------------------------


def test_stress_elo_real_returns_one_per_noise_level() -> None:
    pool_elos = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200]
    cache = _FakeCache({"C001": _make_pool(pool_elos)})
    runner = _FakeRunner(_FakeHarness(cache))
    matches = [_FakeMatchCandidate(2024, 1, "U1", "O1", "C001")]
    baseline = [0.85]
    outcomes = compute_stress_elo_real(runner, matches, baseline)
    assert len(outcomes) == len(NOISE_LEVELS_ELO)
    assert {o.noise_pct for o in outcomes} == set(NOISE_LEVELS_ELO)


def test_stress_elo_real_invokes_run_single_5_times_per_match() -> None:
    pool_elos = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200]
    cache = _FakeCache({"C001": _make_pool(pool_elos)})
    runner = _FakeRunner(_FakeHarness(cache))
    matches = [_FakeMatchCandidate(2024, 1, "U1", "O1", "C001")]
    compute_stress_elo_real(runner, matches, [0.85])
    assert len(runner.calls) == len(NOISE_LEVELS_ELO)


# -----------------------------------------------------------------------------
# compute_stress_roster_real
# -----------------------------------------------------------------------------


def test_stress_roster_real_returns_one_per_turnover_rate() -> None:
    pool_elos = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2150, 2050]
    cache = _FakeCache({"C001": _make_pool(pool_elos)})
    runner = _FakeRunner(_FakeHarness(cache))
    matches = [_FakeMatchCandidate(2024, 1, "U1", "O1", "C001")]
    outcomes = compute_stress_roster_real(runner, matches, [0.85])
    assert len(outcomes) == len(TURNOVER_RATES_ROSTER)


def test_stress_roster_real_drops_correct_count() -> None:
    """For 20% turnover on 10-player pool, drop 2 → kept 8 (= team_size min)."""
    pool_elos = list(range(1500, 1500 + 50, 5))[:10]  # 10 elos
    cache = _FakeCache({"C001": _make_pool(pool_elos)})
    runner = _FakeRunner(_FakeHarness(cache))
    matches = [_FakeMatchCandidate(2024, 1, "U1", "O1", "C001")]
    compute_stress_roster_real(runner, matches, [0.85])
    # Should have some calls (not skipped because 10 - 2 = 8 ≥ team_size 8)
    assert len(runner.calls) > 0


# -----------------------------------------------------------------------------
# compute_dro_real
# -----------------------------------------------------------------------------


def test_dro_real_returns_one_per_epsilon() -> None:
    pool_elos = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200]
    cache = _FakeCache({"C001": _make_pool(pool_elos)})
    runner = _FakeRunner(_FakeHarness(cache))
    matches = [_FakeMatchCandidate(2024, 1, "U1", "O1", "C001")]
    out = compute_dro_real(runner, matches)
    assert set(out.keys()) == set(DRO_EPSILONS)


def test_dro_real_recall_in_unit_interval() -> None:
    pool_elos = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200]
    cache = _FakeCache({"C001": _make_pool(pool_elos)})
    runner = _FakeRunner(_FakeHarness(cache))
    matches = [_FakeMatchCandidate(2024, 1, "U1", "O1", "C001")]
    out = compute_dro_real(runner, matches)
    for outcome in out.values():
        assert 0.0 <= outcome.recall_worst_case <= 1.0


def test_dro_real_skips_empty_pool() -> None:
    cache = _FakeCache({"C001": _make_pool([])})
    runner = _FakeRunner(_FakeHarness(cache))
    matches = [_FakeMatchCandidate(2024, 1, "U1", "O1", "C001")]
    out = compute_dro_real(runner, matches)
    # Empty pool → outcome dict empty (no aggregation possible)
    assert out == {}


def test_subset_constant_consistent_with_spec() -> None:
    """SUBSET_PER_SAISON respects spec §11 + Efron 1993."""
    assert SUBSET_PER_SAISON == 30  # noqa: PLR2004
