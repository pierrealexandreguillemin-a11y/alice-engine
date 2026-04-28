"""Tests T11 BacktestRunner — smoke test on real FFE data.

ISO 29119 : integration test authentique (harness -> ML -> metrics -> report).
Scope : 5 matches hold-out 2024 pour valider le wiring end-to-end.
Gate validation exhaustive = T11 Kaggle adapter (full 100+ matches).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.backtest.harness import BacktestHarness
from scripts.backtest.runner import (
    BacktestReport,
    BacktestRunner,
    MatchStats,
    RunnerConfig,
)

J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")

pytestmark = pytest.mark.skipif(
    not (J.exists() and E.exists()),
    reason="data parquets absent",
)


@pytest.fixture(scope="module")
def harness() -> BacktestHarness:
    """Module-scoped harness (setup once)."""
    h = BacktestHarness()
    h.setup()
    return h


def test_runner_sample_matches_returns_viable_pairs(harness: BacktestHarness) -> None:
    """sample_matches dedupe + team_to_club resolves (returns MatchCandidate)."""
    runner = BacktestRunner(harness=harness, config=RunnerConfig(max_matches=10))
    matches = runner.sample_matches()
    assert len(matches) > 0
    for cand in matches:
        assert isinstance(cand.opp_club, str)
        assert cand.opp_club


def test_runner_run_produces_report_smoke(harness: BacktestHarness) -> None:
    """Run 3-5 matches, verify BacktestReport structure."""
    runner = BacktestRunner(
        harness=harness,
        config=RunnerConfig(max_matches=5, rondes=(5,), n_bootstrap=200),
    )
    report = runner.run()
    assert isinstance(report, BacktestReport)
    assert report.n_matches >= 2
    assert len(report.per_match) == report.n_matches
    assert all(isinstance(s, MatchStats) for s in report.per_match)
    # CIs valid bounds
    for ci in (
        report.ci_recall,
        report.ci_accuracy,
        report.ci_jaccard,
        report.ci_brier,
        report.ci_ece,
    ):
        assert ci.lower <= ci.point <= ci.upper
    # Gates summary returns dict with 7 gates (P3G07..P3G11 + MAE)
    gates = report.gates_summary()
    assert len(gates) == 7
    assert "P3G11_mae" in gates
    assert all(isinstance(v, bool) for v in gates.values())
    # T12 MAE finite value
    assert report.ci_mae.lower >= 0.0
    # Each match has e_score fields
    for s in report.per_match:
        assert s.e_score_mae >= 0.0


def test_runner_raises_if_too_few_matches(harness: BacktestHarness) -> None:
    """< 2 completed matches → ValueError (bootstrap constraint)."""
    runner = BacktestRunner(
        harness=harness,
        config=RunnerConfig(max_matches=0),
    )
    with pytest.raises(ValueError, match="matches for bootstrap"):
        runner.run()
