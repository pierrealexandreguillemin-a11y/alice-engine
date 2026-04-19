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

from services.ffe.rule_engine import Rule, RuleEngine

REAL_A02 = Path("config/ffe_rules/a02.json")
MINI = Path(__file__).parent / "fixtures" / "ffe_rules" / "mini_a02.json"


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
