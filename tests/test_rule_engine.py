"""Tests for services.ffe.rule_engine.

ISO 29119 structure : fixtures + per-UUID assertions.
ISO 5259 : validate lineage_hash determinism.

Document ID: ALICE-TEST-RULE-ENGINE
Version: 1.0.0
"""

from __future__ import annotations

import hashlib  # noqa: F401  (imported per spec for future extensions)
import json
from pathlib import Path

import pytest  # noqa: F401  (imported per spec; pytest discovery)

from services.ali.types import CompetitionContext, PlayerCandidate
from services.ffe.rule_engine import Rule, RuleEngine

REAL_A02 = Path("config/ffe_rules/a02.json")
MINI = Path(__file__).parent / "fixtures" / "ffe_rules" / "mini_a02.json"


def _ctx() -> CompetitionContext:
    return CompetitionContext(
        competition_code="A02",
        niveau="N2",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
    )


def _player(
    nr: str,
    elo: int,
    mute: bool = False,
    licence_active: bool = True,
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
        licence_active=licence_active,
    )


def test_rule_engine_loads_real_a02() -> None:
    engine = RuleEngine.from_json_file(REAL_A02)
    assert len(engine.rules) == 14
    ids = {r.id for r in engine.rules}
    assert "N1-N4_3.7.a_001" in ids


def test_rule_engine_loads_mini_fixture() -> None:
    engine = RuleEngine.from_json_file(MINI)
    assert len(engine.rules) == 1
    r = engine.rules[0]
    assert isinstance(r, Rule)
    assert r.uuid == "TEST_001"


def test_lineage_hash_is_deterministic() -> None:
    e1 = RuleEngine.from_json_file(REAL_A02)
    e2 = RuleEngine.from_json_file(REAL_A02)
    assert e1.lineage_hash() == e2.lineage_hash()
    assert len(e1.lineage_hash()) == 64


def test_lineage_hash_changes_with_content(tmp_path: Path) -> None:
    f = tmp_path / "x.json"
    base = json.loads(MINI.read_text(encoding="utf-8"))
    f.write_text(json.dumps(base), encoding="utf-8")
    e1 = RuleEngine.from_json_file(f)
    base["rules"][0]["texte"] = "autre"
    f.write_text(json.dumps(base), encoding="utf-8")
    e2 = RuleEngine.from_json_file(f)
    assert e1.lineage_hash() != e2.lineage_hash()


def test_filter_candidates_returns_list() -> None:
    engine = RuleEngine.from_json_file(REAL_A02)
    pool = [_player("A1", 2000), _player("A2", 1500)]
    out = engine.filter_candidates(pool, _ctx())
    assert isinstance(out, list)
    assert len(out) <= len(pool)


def test_validate_lineup_empty_returns_team_size_violation() -> None:
    engine = RuleEngine.from_json_file(REAL_A02)
    violations = engine.validate_lineup([], _ctx())
    assert any(v.rule_article == "3.7.a" for v in violations)


def test_validate_lineup_happy_path() -> None:
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player(f"X{i}", 2000 - i * 50) for i in range(8)]
    violations = engine.validate_lineup(lineup, _ctx())
    hard_errors = [v for v in violations if v.severity == "error"]
    assert not any(v.rule_article == "3.7.a" for v in hard_errors)


def _lineup_ok():
    return [_player(f"X{i}", 2000 - i * 50) for i in range(8)]


def test_rule_3_7_a_too_few():
    engine = RuleEngine.from_json_file(REAL_A02)
    violations = engine.validate_lineup(_lineup_ok()[:7], _ctx())
    assert any(v.rule_article == "3.7.a" for v in violations)


def test_rule_3_7_a_too_many():
    engine = RuleEngine.from_json_file(REAL_A02)
    violations = engine.validate_lineup(_lineup_ok() + [_player("X8", 1000)], _ctx())
    assert any(v.rule_article == "3.7.a" for v in violations)


def test_rule_3_6_e_bad_order():
    engine = RuleEngine.from_json_file(REAL_A02)
    # Ecart > 100 Elo entre 2 boards consecutifs = violation
    lineup = [_player("X0", 1600)] + [_player(f"X{i}", 2500) for i in range(1, 8)]
    violations = engine.validate_lineup(lineup, _ctx())
    assert any(v.rule_article == "3.6.e" for v in violations)


def test_rule_3_7_g_too_many_mutes():
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player(f"X{i}", 2000 - i * 50, mute=(i < 4)) for i in range(8)]
    violations = engine.validate_lineup(lineup, _ctx())
    assert any(v.rule_article == "3.7.g" for v in violations)


def test_rule_3_7_j_elo_max():
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player(f"X{i}", 2500) for i in range(8)]
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N4",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=2400,
    )
    violations = engine.validate_lineup(lineup, ctx)
    assert any(v.rule_article == "3.7.j" for v in violations)


def test_rule_3_7_j_not_applied_when_no_cap():
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player(f"X{i}", 2500) for i in range(8)]
    violations = engine.validate_lineup(lineup, _ctx())  # elo_max=None
    assert not any(v.rule_article == "3.7.j" for v in violations)


# ---------------------------------------------------------------------------
# D-P3-11 Plan 2 Task 9 : new articles (3.7.c, 3.7.d, 3.7.e, 3.7.f, 3.7.h, 3.7.i)
# + filter_by_article + check_unique_assignment
# ---------------------------------------------------------------------------


def _ctx_n3_ronde(ronde: int = 3) -> CompetitionContext:
    """N3 context, no fr_gender requirement."""
    return CompetitionContext(
        competition_code="A02",
        niveau="N3",
        ronde=ronde,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
    )


def _player_full(
    nr: str,
    elo: int = 1900,
    matchs_joues: int = 0,
    matchs_equipe_sup: tuple[tuple[str, int, int], ...] | None = None,
    group_history: str | None = None,
    is_french_eu: bool = True,
    is_french: bool = True,
    sexe: str = "M",
    mute: bool = False,
) -> PlayerCandidate:
    """Full PlayerCandidate with all Plan 2 extension fields."""
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
        matchs_joues=matchs_joues,
        matchs_equipe_sup=matchs_equipe_sup,
        group_history=group_history,
        is_french_eu=is_french_eu,
        is_french=is_french,
        sexe=sexe,
    )


def test_rule_3_7_c_brule():
    """3.7.c : player with matchs_equipe_sup >= seuil in team rank < target -> violation."""
    engine = RuleEngine.from_json_file(REAL_A02)
    # target team rank 2, player burned on stronger team (rank 1) with 3 matches
    lineup = [_player_full(f"X{i}", 2000 - i * 50) for i in range(7)]
    burned = _player_full("X7", 1650, matchs_equipe_sup=(("CLUB_1", 3, 1),))
    lineup.append(burned)
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N3",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
        target_team_id="CLUB_2",
        target_team_rank=2,
    )
    violations = engine.validate_lineup(lineup, ctx)
    assert any(v.rule_article == "3.7.c" for v in violations)


def test_rule_3_7_c_brule_same_team_ok():
    """3.7.c : matches played in the SAME team don't burn."""
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [
        _player_full(f"X{i}", 2000 - i * 50, matchs_equipe_sup=(("CLUB_2", 5, 2),))
        for i in range(8)
    ]
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N3",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
        target_team_id="CLUB_2",
        target_team_rank=2,
    )
    violations = engine.validate_lineup(lineup, ctx)
    assert not any(v.rule_article == "3.7.c" for v in violations)


def test_rule_3_7_e_match_count():
    """3.7.e : player with matchs_joues >= ronde -> violation."""
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player_full(f"X{i}", 2000 - i * 50, matchs_joues=0) for i in range(7)]
    lineup.append(_player_full("X7", 1650, matchs_joues=3))  # exceeds ronde=3
    violations = engine.validate_lineup(lineup, _ctx_n3_ronde(ronde=3))
    assert any(v.rule_article == "3.7.e" for v in violations)


def test_rule_3_7_d_same_group():
    """3.7.d : player with group_history != target_group -> violation."""
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player_full(f"X{i}", 2000 - i * 50) for i in range(7)]
    lineup.append(_player_full("X7", 1650, group_history="groupA"))
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N3",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
        target_group="groupB",
    )
    violations = engine.validate_lineup(lineup, ctx)
    assert any(v.rule_article == "3.7.d" for v in violations)


def test_rule_3_7_f_noyau_ronde_1_passes():
    """3.7.f : ronde 1 -> noyau check always passes."""
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player_full(f"X{i}", 2000 - i * 50) for i in range(8)]
    violations = engine.validate_lineup(lineup, _ctx_n3_ronde(ronde=1))
    assert not any(v.rule_article == "3.7.f" for v in violations)


def test_rule_3_7_f_noyau_ronde_2_violates():
    """3.7.f : ronde >= 2 with <50% in noyau -> violation."""
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player_full(f"X{i}", 2000 - i * 50) for i in range(8)]
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N3",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
        noyau=frozenset({"X0", "X1", "X2"}),  # only 3/8 in noyau
    )
    violations = engine.validate_lineup(lineup, ctx)
    assert any(v.rule_article == "3.7.f" for v in violations)


def test_rule_3_7_f_noyau_ronde_2_passes():
    """3.7.f : ronde >= 2 with >=50% in noyau -> no violation."""
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player_full(f"X{i}", 2000 - i * 50) for i in range(8)]
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N3",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
        noyau=frozenset({"X0", "X1", "X2", "X3", "X4"}),  # 5/8 in noyau
    )
    violations = engine.validate_lineup(lineup, ctx)
    assert not any(v.rule_article == "3.7.f" for v in violations)


def test_rule_3_7_h_foreign_quota():
    """3.7.h : <5 FR/UE -> violation."""
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player_full(f"X{i}", 2000 - i * 50, is_french_eu=(i < 4)) for i in range(8)]
    ctx = _ctx_n3_ronde(ronde=3)
    violations = engine.validate_lineup(lineup, ctx)
    assert any(v.rule_article == "3.7.h" for v in violations)


def test_rule_3_7_i_fr_gender_n1_requires_fr_male_female():
    """3.7.i : N1 without 1 FR male + 1 FR female -> violation."""
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player_full(f"X{i}", 2000 - i * 50) for i in range(8)]  # all FR male
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N1",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
    )
    violations = engine.validate_lineup(lineup, ctx)
    assert any(v.rule_article == "3.7.i" for v in violations)


def test_rule_3_7_i_fr_gender_n1_passes_with_mixed():
    """3.7.i : N1 with 1 FR male + 1 FR female -> no violation."""
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player_full(f"X{i}", 2000 - i * 50) for i in range(7)]
    lineup.append(_player_full("X7", 1650, sexe="F"))
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N1",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
    )
    violations = engine.validate_lineup(lineup, ctx)
    assert not any(v.rule_article == "3.7.i" for v in violations)


def test_rule_3_7_i_fr_gender_n3_no_constraint():
    """3.7.i : Not N1/N2/Top16 -> no check."""
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player_full(f"X{i}", 2000 - i * 50) for i in range(8)]  # all FR male
    violations = engine.validate_lineup(lineup, _ctx_n3_ronde(ronde=3))
    assert not any(v.rule_article == "3.7.i" for v in violations)


def test_check_unique_assignment_no_duplicates():
    """1 joueur = 1 equipe (cross-team OK)."""
    assert RuleEngine.check_unique_assignment([["A1", "A2"], ["A3", "A4"]]) is True


def test_check_unique_assignment_duplicate():
    """1 joueur = 1 equipe (cross-team duplicate)."""
    assert RuleEngine.check_unique_assignment([["A1", "A2"], ["A2", "A3"]]) is False


def test_filter_by_article_brule():
    """filter_by_article('3.7.c') removes burned players."""
    engine = RuleEngine.from_json_file(REAL_A02)
    pool = [
        _player_full("OK", 1900),
        _player_full("BAD", 2000, matchs_equipe_sup=(("CLUB_1", 3, 1),)),
    ]
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N3",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
        target_team_id="CLUB_2",
        target_team_rank=2,
    )
    out = engine.filter_by_article(pool, "3.7.c", ctx)
    assert len(out) == 1
    assert out[0].nr_ffe == "OK"


def test_filter_by_article_match_count():
    """filter_by_article('3.7.e') removes players at quota."""
    engine = RuleEngine.from_json_file(REAL_A02)
    pool = [
        _player_full("OK", 1900, matchs_joues=1),
        _player_full("BAD", 2000, matchs_joues=3),
    ]
    out = engine.filter_by_article(pool, "3.7.e", _ctx_n3_ronde(ronde=3))
    assert len(out) == 1
    assert out[0].nr_ffe == "OK"


def test_filter_by_article_same_group():
    """filter_by_article('3.7.d') removes players with different group_history."""
    engine = RuleEngine.from_json_file(REAL_A02)
    pool = [
        _player_full("OK", 1900, group_history="groupA"),
        _player_full("BAD", 2000, group_history="groupB"),
        _player_full("NEW", 1800, group_history=None),  # untouched
    ]
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N3",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
        target_group="groupA",
    )
    out = engine.filter_by_article(pool, "3.7.d", ctx)
    assert {p.nr_ffe for p in out} == {"OK", "NEW"}


def test_filter_by_article_elo_max():
    """filter_by_article('3.7.j') removes over-Elo players."""
    engine = RuleEngine.from_json_file(REAL_A02)
    pool = [_player_full("OK", 2300), _player_full("BAD", 2500)]
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N4",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=2400,
    )
    out = engine.filter_by_article(pool, "3.7.j", ctx)
    assert len(out) == 1
    assert out[0].nr_ffe == "OK"


def test_filter_by_article_unknown_returns_pool():
    """Unknown article returns pool unchanged."""
    engine = RuleEngine.from_json_file(REAL_A02)
    pool = [_player_full("A", 1900), _player_full("B", 2000)]
    out = engine.filter_by_article(pool, "99.9.x", _ctx_n3_ronde())
    assert len(out) == 2
