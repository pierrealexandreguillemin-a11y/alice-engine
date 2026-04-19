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

from services.ali.types import CompetitionContext, PlayerCandidate, RuleViolation
from services.ffe.schemas import RuleModel, RulesDocument

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


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
        applied in validate_lineup.
        """
        out = list(pool)
        for rule in self._rules:
            if rule.effet != "restrict_team_composition":
                continue
            if rule.article == "3.7.j" and context.elo_max is not None:
                out = [p for p in out if p.elo <= context.elo_max]
        return out

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
            v = self._check_rule(rule, lineup, context)
            if v is not None:
                violations.append(v)
        return violations

    def _check_rule(
        self,
        rule: Rule,
        lineup: list[PlayerCandidate],
        context: CompetitionContext,
    ) -> RuleViolation | None:
        """Dispatcher per-article check. Returns RuleViolation or None.

        Handles articles 3.7.a (team_size), 3.6.e (elo ordering),
        3.7.g (max mutes), 3.7.j (elo_max). Each checker is rank A.
        """
        checkers: dict[
            str,
            Callable[[Rule, list[PlayerCandidate], CompetitionContext], RuleViolation | None],
        ] = {
            "3.7.a": self._check_team_size,
            "3.6.e": self._check_elo_order,
            "3.7.g": self._check_mutes_limit,
            "3.7.j": self._check_elo_max,
        }
        checker = checkers.get(rule.article)
        if checker is None:
            return None
        return checker(rule, lineup, context)

    @staticmethod
    def _check_team_size(
        rule: Rule,
        lineup: list[PlayerCandidate],
        context: CompetitionContext,
    ) -> RuleViolation | None:
        """Article 3.7.a : lineup length matches expected team_size."""
        expected_raw = rule.conditions.get("team_size", context.team_size)
        expected = int(expected_raw) if expected_raw is not None else context.team_size
        if len(lineup) != expected:
            return RuleViolation(
                rule_uuid=rule.uuid,
                rule_article=rule.article,
                message=f"team_size: expected {expected}, got {len(lineup)}",
                severity="error",
            )
        return None

    @staticmethod
    def _check_elo_order(
        rule: Rule,
        lineup: list[PlayerCandidate],
        context: CompetitionContext,  # noqa: ARG004  (dispatcher signature parity)
    ) -> RuleViolation | None:
        """Article 3.6.e : Elo descending across boards within tolerance."""
        tolerance_raw = rule.conditions.get("elo_tolerance", 100)
        tolerance = int(tolerance_raw)
        elos = [p.elo for p in lineup]
        for i in range(len(elos) - 1):
            if elos[i + 1] - elos[i] > tolerance:
                return RuleViolation(
                    rule_uuid=rule.uuid,
                    rule_article=rule.article,
                    message=(
                        f"elo order: board {i + 1}={elos[i]} "
                        f"< board {i + 2}={elos[i + 1]} + tol"
                    ),
                    severity="error",
                )
        return None

    @staticmethod
    def _check_mutes_limit(
        rule: Rule,
        lineup: list[PlayerCandidate],
        context: CompetitionContext,
    ) -> RuleViolation | None:
        """Article 3.7.g : number of muted players within allowed max."""
        max_m_raw = rule.conditions.get("max_mutes", context.max_mutes)
        max_m = int(max_m_raw) if max_m_raw is not None else context.max_mutes
        muted = sum(1 for p in lineup if p.mute)
        if muted > max_m:
            return RuleViolation(
                rule_uuid=rule.uuid,
                rule_article=rule.article,
                message=f"mutes: {muted} > max {max_m}",
                severity="error",
            )
        return None

    @staticmethod
    def _check_elo_max(
        rule: Rule,
        lineup: list[PlayerCandidate],
        context: CompetitionContext,
    ) -> RuleViolation | None:
        """Article 3.7.j : no player above optional elo_max cap."""
        if context.elo_max is None:
            return None
        over = [p for p in lineup if p.elo > context.elo_max]
        if over:
            return RuleViolation(
                rule_uuid=rule.uuid,
                rule_article=rule.article,
                message=f"elo_max: {len(over)} players above {context.elo_max}",
                severity="error",
            )
        return None
