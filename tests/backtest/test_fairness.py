"""Tests T13 Smoke fairness breakdown — Plan 3 V2 Phase 3 (ISO 24027)."""

from __future__ import annotations

import pytest

from scripts.backtest.fairness import (
    FAIRNESS_RECALL_GATE,
    GroupStats,
    breakdown_by_key,
    breakdown_by_opponent_club_size,
    breakdown_by_ronde,
    gates_per_group,
    max_gap,
)
from scripts.backtest.runner_types import MatchStats


def _mk_match(ronde: int = 5, recall: float = 0.9, mae: float = 0.5) -> MatchStats:
    return MatchStats(
        saison=2024,
        ronde=ronde,
        user_team="U",
        opponent_team="O",
        recall_ali=recall,
        accuracy_ali=recall,
        jaccard_ali=recall,
        brier_ali=0.1,
        ece_ali=0.03,
        recall_baseline=0.8,
        brier_baseline=0.2,
        bss=0.1,
        e_score_predicted=4.0,
        e_score_observed=4.0,
        e_score_mae=mae,
        ali_correct=recall >= 0.9,
        baseline_correct=False,
    )


def test_breakdown_by_key_groups_correctly():
    matches = [_mk_match(ronde=5), _mk_match(ronde=5), _mk_match(ronde=9)]
    bd = breakdown_by_key(matches, key_fn=lambda m: f"r{m.ronde}")
    assert set(bd.keys()) == {"r5", "r9"}
    assert bd["r5"].n == 2
    assert bd["r9"].n == 1


def test_breakdown_by_ronde_stratifies():
    matches = [_mk_match(ronde=r) for r in (5, 5, 7, 9, 9, 9)]
    bd = breakdown_by_ronde(matches)
    assert bd["ronde_5"].n == 2
    assert bd["ronde_9"].n == 3


def test_breakdown_means_correct():
    matches = [_mk_match(ronde=5, recall=0.8), _mk_match(ronde=5, recall=1.0)]
    bd = breakdown_by_ronde(matches)
    assert bd["ronde_5"].recall_mean == pytest.approx(0.9)


def test_breakdown_by_pool_size_quartiles():
    # 8 matches avec pool sizes 10, 20, 30, 40, 50, 60, 70, 80
    matches = [_mk_match() for _ in range(8)]
    sizes = {id(m): 10 + i * 10 for i, m in enumerate(matches)}
    bd = breakdown_by_opponent_club_size(matches, pool_size_fn=lambda m: sizes[id(m)])
    assert "small" in bd
    assert "large" in bd or "xlarge" in bd


def test_gates_per_group_passes_above_threshold():
    matches = [_mk_match(recall=0.92)]
    bd = breakdown_by_ronde(matches)
    gates = gates_per_group(bd, gate=FAIRNESS_RECALL_GATE)
    assert all(gates.values())


def test_gates_per_group_fails_below_threshold():
    matches = [_mk_match(recall=0.50)]
    bd = breakdown_by_ronde(matches)
    gates = gates_per_group(bd, gate=FAIRNESS_RECALL_GATE)
    assert not any(gates.values())


def test_max_gap_equal_groups_returns_zero():
    matches = [_mk_match(recall=0.9), _mk_match(ronde=9, recall=0.9)]
    bd = breakdown_by_ronde(matches)
    assert max_gap(bd) == pytest.approx(0.0)


def test_max_gap_unequal_groups():
    matches = [_mk_match(ronde=5, recall=0.9), _mk_match(ronde=9, recall=0.6)]
    bd = breakdown_by_ronde(matches)
    assert max_gap(bd) == pytest.approx(0.3)


def test_group_stats_frozen():
    matches = [_mk_match()]
    bd = breakdown_by_ronde(matches)
    g = next(iter(bd.values()))
    assert isinstance(g, GroupStats)
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        g.n = 99  # type: ignore[misc]


def test_breakdown_empty_matches():
    bd = breakdown_by_ronde([])
    assert bd == {}
    assert max_gap(bd) == 0.0
