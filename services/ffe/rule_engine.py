"""FFE RuleEngine - generic JSON-driven rules interpreter.

ISO 5055 : SRP, module < 300 lignes. Dispatchers delegate to
services.ffe.checkers (lineup validation) and services.ffe.filters
(pre-composition eligibility).
ISO 42001 : traceability via UUID + source_ref per rule.
ISO 5259 : lineage_hash SHA-256 of loaded JSON.
ISO 27034 : Pydantic validation at load.

Replaces services/ffe_rules.py (ADR-013, ADR-015 migration Plan 2 Task 9).
Covers 10 articles : 3.6.e, 3.7.a, 3.7.c, 3.7.d, 3.7.e, 3.7.f, 3.7.g,
3.7.h, 3.7.i, 3.7.j.

Document ID: ALICE-FFE-RULE-ENGINE
Version: 2.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from services.ffe import checkers as chk
from services.ffe import filters as flt
from services.ffe.schemas import RuleModel, RulesDocument

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from services.ali.types import CompetitionContext, PlayerCandidate, RuleViolation


# Dispatcher maps article -> checker function.
_CHECKERS: dict[
    str,
    Callable[[Rule, list[PlayerCandidate], CompetitionContext], RuleViolation | None],
] = {
    "3.7.a": chk.check_team_size,
    "3.6.e": chk.check_elo_order,
    "3.7.c": chk.check_brule,
    "3.7.d": chk.check_same_group,
    "3.7.e": chk.check_match_count,
    "3.7.f": chk.check_noyau,
    "3.7.g": chk.check_mutes_limit,
    "3.7.h": chk.check_foreign_quota,
    "3.7.i": chk.check_fr_gender,
    "3.7.j": chk.check_elo_max,
}


@dataclass(frozen=True)
class Rule:
    """One FFE rule, domain dataclass (frozen for ISO 29119 testability).

    ISO 42001 : `uuid_rfc4122` = canonical UUID used cross-system for rule
    traceability (joined with audit logs and chess-app). `uuid` = human id.
    """

    uuid: str
    uuid_rfc4122: str
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
            uuid_rfc4122=model.uuid_rfc4122,
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
        pool = engine.filter_by_article(pool, "3.7.c", context)
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

    def filter_candidates(
        self,
        pool: list[PlayerCandidate],
        context: CompetitionContext,
    ) -> list[PlayerCandidate]:
        """Filter pool with rules that restrict player ELIGIBILITY.

        Currently applies article 3.7.j (elo_max if set). Most eligibility
        rules require external data (brule history, same-group) and are
        applied via filter_by_article.
        """
        out = list(pool)
        for rule in self._rules:
            if rule.effet != "restrict_team_composition":
                continue
            if rule.article == "3.7.j" and context.elo_max is not None:
                out = [p for p in out if p.elo <= context.elo_max]
        return out

    def filter_by_article(
        self,
        pool: list[PlayerCandidate],
        article: str,
        context: CompetitionContext,
    ) -> list[PlayerCandidate]:
        """Eligibility filter par article (D-P3-11 Plan 2 migration).

        Supports 3.7.c (brule), 3.7.d (same_group), 3.7.e (match_count),
        3.7.j (elo_max). Returns pool unchanged if article unsupported.
        """
        if article == "3.7.c":
            return flt.filter_brule(pool, context)
        if article == "3.7.d":
            return flt.filter_same_group(pool, context)
        if article == "3.7.e":
            return flt.filter_match_count(pool, context)
        if article == "3.7.j":
            return flt.filter_elo_max(pool, context)
        return list(pool)

    @staticmethod
    def check_unique_assignment(teams: list[list[str]]) -> bool:
        """1 joueur = 1 equipe par match (cross-team helper)."""
        return flt.check_unique_assignment(teams)

    def validate_lineup(
        self,
        lineup: list[PlayerCandidate],
        context: CompetitionContext,
    ) -> list[RuleViolation]:
        """Validate a full lineup against all composition rules."""
        violations: list[RuleViolation] = []
        for rule in self._rules:
            if rule.effet != "restrict_team_composition":
                continue
            checker = _CHECKERS.get(rule.article)
            if checker is None:
                continue
            v = checker(rule, lineup, context)
            if v is not None:
                violations.append(v)
        return violations
