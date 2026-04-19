"""Tests for FFE Pydantic schemas (Phase 3 Plan 1 Task 5).

ISO 27034 : input validation.
ISO 25012 : data quality structural schema.

Document ID: ALICE-FFE-SCHEMAS-TEST
Version: 1.0.0
Count: 4
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from services.ffe.schemas import RuleModel, RulesDocument

FIXTURE = Path(__file__).parent / "fixtures" / "ffe_rules" / "mini_a02.json"


def test_load_mini_a02() -> None:
    """Mini fixture loads and validates as RulesDocument."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    doc = RulesDocument.model_validate(data)
    assert len(doc.rules) == 1
    assert doc.rules[0].uuid == "TEST_001"
    assert doc.rules[0].article == "3.7.a"


def test_load_real_a02() -> None:
    """Real config/ffe_rules/a02.json loads (14 rules)."""
    data = json.loads(Path("config/ffe_rules/a02.json").read_text(encoding="utf-8"))
    doc = RulesDocument.model_validate(data)
    assert len(doc.rules) == 14


def test_rejects_missing_uuid() -> None:
    """Missing uuid field raises ValidationError."""
    bad = {
        "uuid_rfc4122": "x",
        "id": "i",
        "source_ref": "r",
        "document": "d",
        "version": "v",
        "article": "3.7.a",
        "texte": "t",
        "conditions": {},
        "effet": "restrict_team_composition",
        "exceptions": [],
        "priority": 1,
        "date_version": "2025-07-01",
    }
    with pytest.raises(ValidationError):
        RuleModel.model_validate(bad)


def test_rejects_invalid_effet() -> None:
    """Invalid effet value (outside Literal) raises ValidationError."""
    bad = {
        "uuid": "U",
        "uuid_rfc4122": "x",
        "id": "i",
        "source_ref": "r",
        "document": "d",
        "version": "v",
        "article": "a",
        "texte": "t",
        "conditions": {},
        "effet": "invalid_effet",
        "exceptions": [],
        "priority": 1,
        "date_version": "2025-07-01",
    }
    with pytest.raises(ValidationError):
        RuleModel.model_validate(bad)
