"""Tests for scripts/backtest/pilot_phase4a.py + helpers (Phase 4a T9.5).

Pure-function unit tests ONLY : no harness setup, no ML model load, no real
pilot run. Covers the early-gate decision boundary, the viable/non-viable
candidate trio (target-present + >=1-superior), and the paired statistics
aggregation (empty-guard + populated case).

Document ID: ALICE-BACKTEST-PILOT-PHASE4A-TEST
Version: 1.0.0
Count: 9 unit tests — pure-function, inline payloads only.
"""

from __future__ import annotations

from types import SimpleNamespace

from scripts.backtest.clubs_teams_fixture import build_team_to_club_index
from scripts.backtest.pilot_phase4a import (
    EARLY_GATE_RECALL,
    _viable_sim,
    early_gate_decision,
    names_index,
)
from scripts.backtest.pilot_phase4a_helpers import aggregate_stats

_PAYLOAD = {
    "clubs": {
        "X": {
            "rondes": {
                "3": [
                    ["X 1", "Nationale 1", 8, "2024-11-17"],
                    ["X 3", "Nationale 3", 8, "2024-11-17"],
                ]
            }
        }
    }
}


def _idx() -> dict[str, str]:
    return build_team_to_club_index(_PAYLOAD)


# --- Early-gate boundary -------------------------------------------------


def test_gate_pass_at_threshold() -> None:
    """PASS at exactly the threshold (>= is inclusive)."""
    assert early_gate_decision(EARLY_GATE_RECALL).startswith("PASS")


def test_gate_fail_below_threshold() -> None:
    """FAIL strictly below the threshold."""
    assert early_gate_decision(EARLY_GATE_RECALL - 0.01).startswith("FAIL")


# --- Viability contract (target present + >=1 superior) ------------------


def test_viable_when_target_has_superior() -> None:
    """X 3 (N3) is preceded by X 1 (N1) -> viable, 1 superior team."""
    cand = SimpleNamespace(opp_team="X 3", ronde=3, date="2024-11-17")
    sim = _viable_sim(_PAYLOAD, _idx(), cand)
    assert sim is not None
    assert names_index(sim, "X 3") == 1


def test_non_viable_when_no_superior() -> None:
    """X 1 is team_1 (rank 0) -> no superior team -> non-viable."""
    cand = SimpleNamespace(opp_team="X 1", ronde=3, date="2024-11-17")
    assert _viable_sim(_PAYLOAD, _idx(), cand) is None


def test_non_viable_when_target_dropped_by_date() -> None:
    """A date with no matching entry drops everything -> non-viable."""
    cand = SimpleNamespace(opp_team="X 3", ronde=3, date="2024-12-25")
    assert _viable_sim(_PAYLOAD, _idx(), cand) is None


def test_non_viable_when_date_missing() -> None:
    """Empty date string cannot be resolved against the fixture -> non-viable."""
    cand = SimpleNamespace(opp_team="X 3", ronde=3, date="")
    assert _viable_sim(_PAYLOAD, _idx(), cand) is None


# --- Paired statistics aggregation ---------------------------------------


def test_aggregate_stats_empty_is_guarded() -> None:
    """Empty rows must not raise (Wilcoxon/McNemar ValueError guarded)."""
    out = aggregate_stats([])
    assert out["mcnemar"] is None
    assert out["wilcoxon"] is None


def test_aggregate_stats_paired() -> None:
    """Populated rows produce wilcoxon keys + a discordant-count for McNemar."""
    rows = [
        {"recall": 0.8, "recall_baseline": 0.5},
        {"recall": 0.6, "recall_baseline": 0.6},
        {"recall": 0.9, "recall_baseline": 0.4},
        {"recall": 0.7, "recall_baseline": 0.55},
    ]
    out = aggregate_stats(rows)
    assert "wilcoxon_p" in out
    assert "wilcoxon_significant" in out
    assert "mcnemar_n_disc" in out


def test_aggregate_stats_all_concordant_mcnemar_guarded() -> None:
    """All pairs concordant (n_disc=0) must not crash; mcnemar_p stays None."""
    rows = [
        {"recall": 0.9, "recall_baseline": 0.9},
        {"recall": 0.2, "recall_baseline": 0.2},
    ]
    out = aggregate_stats(rows)
    assert out["mcnemar_n_disc"] == 0
    assert out["mcnemar_p"] is None
    # Wilcoxon still computable (continuous diffs may be all-zero -> p=1.0).
    assert "wilcoxon_p" in out
