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
