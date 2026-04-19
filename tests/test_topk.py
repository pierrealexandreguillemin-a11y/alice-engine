"""Tests for services.ali.topk — TopKEnumerator deterministe.

ISO 29119 : tests reproductibles, scenarios explicites.
ISO 5055 : SRP, per-assertion clarity.

Document ID: ALICE-TEST-ALI-TOPK
Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.ali.topk import TopKEnumerator
from services.ali.types import CompetitionContext, PlayerCandidate
from services.ffe.rule_engine import RuleEngine

REAL_A02 = Path("config/ffe_rules/a02.json")


def _player(
    nr: str, elo: int, taux: float = 0.8, *, mute: bool = False,
) -> PlayerCandidate:
    return PlayerCandidate(
        nr_ffe=nr,
        nom=f"P{nr}",
        prenom="X",
        elo=elo,
        club="C1",
        mute=mute,
        genre="M",
        categorie="SE",
        licence_active=True,
        taux_presence_effectif=taux,
    )


def _ctx(team_size: int = 8) -> CompetitionContext:
    return CompetitionContext(
        competition_code="A02",
        niveau="N2",
        ronde=3,
        team_size=team_size,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
    )


def test_topk_returns_k_distinct_lineups():
    """Top-K = 3 doit retourner 3 lineups distincts."""
    pool = [_player(f"P{i}", 2200 - i * 30, taux=0.9 - i * 0.05) for i in range(12)]
    engine = RuleEngine.from_json_file(REAL_A02)
    enumerator = TopKEnumerator(engine=engine)
    scenarios = enumerator.enumerate(pool, _ctx(team_size=8), k=3)
    assert len(scenarios) == 3
    sigs = {tuple(a.player.nr_ffe for a in s.lineup.assignments) for s in scenarios}
    assert len(sigs) == 3, "scenarios should be distinct"


def test_topk_lineup_size_matches_team_size():
    pool = [_player(f"P{i}", 2200 - i * 30) for i in range(12)]
    engine = RuleEngine.from_json_file(REAL_A02)
    enumerator = TopKEnumerator(engine=engine)
    scenarios = enumerator.enumerate(pool, _ctx(team_size=8), k=2)
    for s in scenarios:
        assert s.lineup.team_size == 8
        assert len(s.lineup.assignments) == 8


def test_topk_deterministic_same_inputs():
    pool = [_player(f"P{i}", 2200 - i * 30, taux=0.9 - i * 0.05) for i in range(12)]
    engine = RuleEngine.from_json_file(REAL_A02)
    enumerator = TopKEnumerator(engine=engine)
    s1 = enumerator.enumerate(pool, _ctx(team_size=8), k=3)
    s2 = enumerator.enumerate(pool, _ctx(team_size=8), k=3)
    sigs1 = [tuple(a.player.nr_ffe for a in s.lineup.assignments) for s in s1]
    sigs2 = [tuple(a.player.nr_ffe for a in s.lineup.assignments) for s in s2]
    assert sigs1 == sigs2


def test_topk_top_lineup_uses_highest_taux_players():
    """Lineup #1 (top scoring) = joueurs avec plus haut produit P(present) x top boards."""
    pool = [_player(f"P{i}", 2200 - i * 30, taux=0.9 - i * 0.05) for i in range(10)]
    engine = RuleEngine.from_json_file(REAL_A02)
    enumerator = TopKEnumerator(engine=engine)
    scenarios = enumerator.enumerate(pool, _ctx(team_size=8), k=1)
    top_lineup = scenarios[0].lineup
    # Best 8 par taux/Elo combine = P0..P7
    expected_ids = {f"P{i}" for i in range(8)}
    actual_ids = {a.player.nr_ffe for a in top_lineup.assignments}
    assert actual_ids == expected_ids


def test_topk_respects_elo_descending_order():
    pool = [_player(f"P{i}", 2200 - i * 30) for i in range(12)]
    engine = RuleEngine.from_json_file(REAL_A02)
    enumerator = TopKEnumerator(engine=engine)
    scenarios = enumerator.enumerate(pool, _ctx(team_size=8), k=3)
    for s in scenarios:
        elos = [a.player.elo for a in s.lineup.assignments]
        # Sort by board ascending -> Elo should be descending (with tolerance)
        for i in range(len(elos) - 1):
            assert elos[i] >= elos[i + 1] - 100, f"Elo order violated: {elos}"


def test_topk_pool_too_small_raises():
    pool = [_player(f"P{i}", 2200 - i * 30) for i in range(5)]
    engine = RuleEngine.from_json_file(REAL_A02)
    enumerator = TopKEnumerator(engine=engine)
    with pytest.raises(ValueError, match="too small"):
        enumerator.enumerate(pool, _ctx(team_size=8), k=3)


def test_topk_returns_scenarios_with_source_topk():
    pool = [_player(f"P{i}", 2200 - i * 30) for i in range(10)]
    engine = RuleEngine.from_json_file(REAL_A02)
    enumerator = TopKEnumerator(engine=engine)
    scenarios = enumerator.enumerate(pool, _ctx(team_size=8), k=2)
    for s in scenarios:
        assert s.source == "topk"


def test_topk_backtracking_explores_beyond_single_swap():
    """B&B : top-K explore par priority queue DESC sur joint_prob.

    Verifie la propriete cle de l'algo B&B priorise (spec §4.6) :
    - K lineups distincts meme avec beaucoup de reserves disponibles
    - joint_prob DESC strict (priority queue le garantit)
    """
    pool = [_player(f"P{i}", 2200 - i * 30, taux=0.9 - i * 0.05) for i in range(14)]
    engine = RuleEngine.from_json_file(REAL_A02)
    enumerator = TopKEnumerator(engine=engine)
    scenarios = enumerator.enumerate(pool, _ctx(team_size=8), k=5)

    # Les 5 lineups doivent etre distincts par signature canonique (nr_ffe, board)
    sigs = {
        tuple((a.player.nr_ffe, a.board) for a in s.lineup.assignments)
        for s in scenarios
    }
    assert len(sigs) == 5, f"expected 5 distinct lineups, got {len(sigs)}"

    # joint_prob DESC strict (priority queue le garantit)
    probs = [s.joint_prob for s in scenarios]
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1], f"not DESC: {probs}"


def test_topk_respects_rule_engine_constraints_during_backtracking():
    """B&B : neighbors invalides (contraintes FFE violees) sont pruned.

    Avec 6 mutes dans un pool de 12 et max_mutes=3, tout lineup de 8
    contenant > 3 mutes doit etre rejete par le B&B (prune via
    RuleEngine.validate_lineup). Comme le top initial contient deja 6 mutes,
    on ajoute un pool majoritairement non-mute pour tester le prune.
    """
    # 8 non-mutes (plus forts) + 6 mutes (plus faibles) = 14 total
    non_mutes = [
        _player(f"NM{i}", 2200 - i * 20, taux=0.9 - i * 0.02) for i in range(8)
    ]
    mutes = [
        _player(f"M{i}", 2050 - i * 20, taux=0.5, mute=True) for i in range(6)
    ]
    pool = non_mutes + mutes
    engine = RuleEngine.from_json_file(REAL_A02)
    enumerator = TopKEnumerator(engine=engine)
    scenarios = enumerator.enumerate(pool, _ctx(team_size=8), k=5)

    # Tous les lineups retournes doivent respecter max_mutes=3
    for s in scenarios:
        muted = sum(1 for a in s.lineup.assignments if a.player.mute)
        assert muted <= 3, f"scenario has {muted} mutes > 3 (prune failed)"
