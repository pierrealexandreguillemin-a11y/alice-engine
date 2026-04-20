"""Tests T10 baseline Elo — Plan 3 V2."""

from __future__ import annotations

import pytest

from scripts.backtest.baseline_elo import baseline_elo_brier, baseline_elo_scenario_set
from scripts.backtest.ground_truth import ObservedLineup, ObservedPlayer
from services.ali.types import PlayerCandidate


def _mk_player(nr: str, elo: int) -> PlayerCandidate:
    return PlayerCandidate(
        nr_ffe=nr,
        nom=f"P{nr}",
        prenom="X",
        elo=elo,
        club="C1",
        mute=False,
        genre="M",
        categorie="SE",
        licence_active=True,
    )


def test_baseline_returns_single_scenario():
    pool = [_mk_player(f"N{i}", 2000 - i * 30) for i in range(10)]
    ss = baseline_elo_scenario_set(pool, team_size=8)
    assert len(ss.scenarios) == 1
    assert ss.scenarios[0].weight == 1.0


def test_baseline_picks_top_elo():
    pool = [_mk_player(f"N{i}", 2000 - i * 30) for i in range(10)]
    ss = baseline_elo_scenario_set(pool, team_size=3)
    assigns = sorted(ss.scenarios[0].lineup.assignments, key=lambda a: a.board)
    # Top 3 Elo = 2000, 1970, 1940 (indices 0, 1, 2)
    assert assigns[0].player.elo == 2000
    assert assigns[1].player.elo == 1970
    assert assigns[2].player.elo == 1940


def test_baseline_raises_if_pool_too_small():
    pool = [_mk_player("N1", 1800), _mk_player("N2", 1700)]
    with pytest.raises(ValueError, match="too small"):
        baseline_elo_scenario_set(pool, team_size=8)


def test_baseline_elo_brier_computes_value():
    pool = [_mk_player(f"N{i}", 2000 - i * 30) for i in range(10)]
    observed = ObservedLineup(
        club_name="C1",
        saison=2024,
        ronde=5,
        players=(
            ObservedPlayer(joueur_nom="PN0 X", echiquier=1, elo=2000),
            ObservedPlayer(joueur_nom="PN1 X", echiquier=2, elo=1970),
        ),
    )
    brier = baseline_elo_brier(observed, pool, team_size=2)
    # Top 2 Elo == observed → Brier = 0
    assert brier == pytest.approx(0.0)


def test_baseline_elo_brier_mismatch_nonzero():
    """Baseline top Elo != observed → Brier > 0."""
    pool = [_mk_player(f"N{i}", 2000 - i * 30) for i in range(10)]
    # Observed is a random non-top player (PN5)
    observed = ObservedLineup(
        club_name="C1",
        saison=2024,
        ronde=5,
        players=(ObservedPlayer(joueur_nom="PN5 X", echiquier=1, elo=1850),),
    )
    brier = baseline_elo_brier(observed, pool, team_size=1)
    assert brier > 0
