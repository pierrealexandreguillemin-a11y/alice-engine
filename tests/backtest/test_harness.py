"""Tests harness P3-Task 1."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.backtest.harness import BacktestHarness

J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")

pytestmark = pytest.mark.skipif(
    not (J.exists() and E.exists()),
    reason="data parquets absent du runner",
)


def test_harness_setup_loads_all_services() -> None:
    """Setup charge cache + rules + classifier + generator + inference."""
    h = BacktestHarness()
    h.setup()
    assert h.cache is not None
    assert h.rule_engine is not None
    assert h.classifier is not None
    assert h.scenario_generator is not None
    assert h.inference is not None
    # feature_store can be None (best effort)


def test_harness_run_match_returns_complete_result() -> None:
    """Pilote feasibility 1 match : services Plan 1+2 chained correctly."""
    h = BacktestHarness()
    h.setup()

    assert h.cache is not None
    clubs = [c for c, df in h.cache.joueurs_by_club.items() if len(df) >= 12][:2]
    if len(clubs) < 2:
        pytest.skip("besoin de 2 clubs avec >=12 joueurs")
    user_club, opp_club = clubs

    user_players = h.cache.joueurs_by_club[user_club].head(8).to_dict("records")
    user_lineup = [{"ffe_id": str(p["nr_ffe"]), "elo": int(p["elo"] or 1500)} for p in user_players]

    result = h.run_match(
        user_club_id=user_club,
        opponent_club_id=opp_club,
        saison=2024,
        ronde=5,
        nb_rondes_total=11,
        division="N3",
        team_size=8,
        user_lineup=user_lineup,
        strict=False,
    )

    assert len(result.scenario_set.scenarios) == 20
    assert len(result.aggregated_boards) == 8
    assert len(result.lineage_hash) == 64
    assert result.elapsed_ms > 0

    for board in result.aggregated_boards:
        total = board.p_win + board.p_draw + board.p_loss
        assert abs(total - 1.0) < 0.01
        assert 0 <= board.p_win <= 1
