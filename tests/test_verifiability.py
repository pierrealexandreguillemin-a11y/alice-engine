from pathlib import Path

import pytest

from services.ali.verifiability import VerifiabilityClassifier
from services.ffe.rule_engine import Rule, RuleEngine

REAL_A02 = Path("config/ffe_rules/a02.json")
CLASSIF = Path("config/ffe_rules/alice_verifiability.json")


def test_classifier_loads_from_json():
    c = VerifiabilityClassifier.from_json_file(CLASSIF)
    assert len(c.classifications) == 14


def test_is_public_returns_true_for_public_rule():
    c = VerifiabilityClassifier.from_json_file(CLASSIF)
    engine = RuleEngine.from_json_file(REAL_A02)
    rule_3_7_a = next(r for r in engine.rules if r.id == "N1-N4_3.7.a_001")
    assert c.is_public(rule_3_7_a) is True


def test_is_public_returns_false_for_private_rule():
    c = VerifiabilityClassifier.from_json_file(CLASSIF)
    engine = RuleEngine.from_json_file(REAL_A02)
    rule_noyau = next(r for r in engine.rules if r.id == "N1-N3_3.7.f_001")
    assert c.is_public(rule_noyau) is False


def test_partition_rules():
    """D-P3-09 (2026-04-28) : 14 règles A02 = 10 public + 3 private + 1 out_of_scope.

    Article 3.7 (arbitrage) reclassé `out_of_scope` car non-composition,
    distinct de `private` qui suppose règle respectée par adversaire.
    """
    c = VerifiabilityClassifier.from_json_file(CLASSIF)
    engine = RuleEngine.from_json_file(REAL_A02)
    public, private = c.partition_rules(engine.rules)
    assert len(public) == 10
    assert len(private) == 3  # 4 → 3 après reclassement N1-N2_3.7_001 en out_of_scope
    assert all(c.is_public(r) for r in public)
    assert all(not c.is_public(r) for r in private)


def test_partition_is_exhaustive_three_class():
    """D-P3-09 : exhaustivité tri-class public + private + out_of_scope == total."""
    c = VerifiabilityClassifier.from_json_file(CLASSIF)
    engine = RuleEngine.from_json_file(REAL_A02)
    public, private = c.partition_rules(engine.rules)
    out_of_scope_count = sum(
        1
        for r in engine.rules
        if c.classifications.get(r.id) is not None
        and c.classifications[r.id].verifiability == "out_of_scope"
    )
    assert len(public) + len(private) + out_of_scope_count == len(engine.rules)
    assert out_of_scope_count == 1  # règle 3.7 arbitrage


def test_unknown_rule_raises():
    c = VerifiabilityClassifier.from_json_file(CLASSIF)
    unknown = Rule(
        uuid="UNKNOWN_XYZ",
        uuid_rfc4122="00000000-0000-4000-8000-000000000999",
        id="UNKNOWN_ID",
        source_ref="",
        article="",
        texte="",
        conditions={},
        effet="restrict_team_composition",
        priority=1,
    )
    with pytest.raises(KeyError):
        c.is_public(unknown)
