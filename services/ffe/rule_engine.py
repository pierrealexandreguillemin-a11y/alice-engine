"""FFE RuleEngine - generic JSON-driven rules interpreter.

ISO 5055 : SRP, module < 300 lignes.
ISO 42001 : traceability via UUID + source_ref per rule.
ISO 5259 : lineage_hash SHA-256 of loaded JSON.
ISO 27034 : Pydantic validation at load.

Replaces services/ffe_rules.py (ADR-013).

Document ID: ALICE-FFE-RULE-ENGINE
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from services.ffe.schemas import RuleModel, RulesDocument

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class Rule:
    """One FFE rule, domain dataclass (frozen for ISO 29119 testability)."""

    uuid: str
    id: str
    source_ref: str
    article: str
    texte: str
    conditions: dict[str, Any]
    effet: str
    priority: int

    @classmethod
    def from_model(cls, model: RuleModel) -> Rule:
        """Build a Rule from a validated Pydantic RuleModel."""
        return cls(
            uuid=model.uuid,
            id=model.id,
            source_ref=model.source_ref,
            article=model.article,
            texte=model.texte,
            conditions=dict(model.conditions),
            effet=model.effet,
            priority=model.priority,
        )


class RuleEngine:
    """Generic engine that interprets FFE rules loaded from JSON.

    Usage:
        engine = RuleEngine.from_json_file(Path("config/ffe_rules/a02.json"))
        violations = engine.validate_lineup(lineup, context)
        pool = engine.filter_candidates(pool, context)
    """

    def __init__(self, rules: list[Rule], source_sha256: str) -> None:
        """Initialize engine with parsed rules and the SHA-256 of the source JSON."""
        self._rules: tuple[Rule, ...] = tuple(rules)
        self._source_sha256 = source_sha256

    @property
    def rules(self) -> tuple[Rule, ...]:
        """Return the immutable tuple of loaded Rule objects."""
        return self._rules

    def lineage_hash(self) -> str:
        """Return SHA-256 of the JSON source for ISO 5259 lineage."""
        return self._source_sha256

    @classmethod
    def from_json_file(cls, path: Path) -> RuleEngine:
        """Load, validate (Pydantic), and instantiate from a JSON file."""
        raw_bytes = path.read_bytes()
        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        data = json.loads(raw_bytes.decode("utf-8"))
        doc = RulesDocument.model_validate(data)
        rules = [Rule.from_model(m) for m in doc.rules]
        return cls(rules=rules, source_sha256=source_sha256)
