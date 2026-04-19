"""VerifiabilityClassifier — PUBLIC/PRIVATE annotations locales ALICE.

ISO 24027 : fairness classification (distinguer ce qu'on peut vérifier
sur l'adversaire depuis ce qu'on doit supposer).
ISO 42001 : traceability des décisions de classification.

Key utilisé : `rule.id` (stable FFE label, ex. "N1-N4_3.7.a_001"),
PAS `rule.uuid` (code court chess-app).

Document ID: ALICE-ALI-VERIFIABILITY
Version: 1.0.0
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from services.ffe.rule_engine import Rule


Verifiability = Literal["public", "private"]


class VerifiabilityClassifier:
    """Classifie les règles FFE en PUBLIC ou PRIVATE.

    PUBLIC : applicable au générateur Monte-Carlo (vérifiable sur adversaire).
    PRIVATE : supposée respectée par l'adversaire (décisions internes club).

    Key : `rule.id` (identifiant stable FFE article-based).
    """

    def __init__(self, classifications: dict[str, dict[str, Any]]) -> None:
        """Initialize classifier with a dict keyed on `rule.id`."""
        self._classifications: dict[str, dict[str, Any]] = dict(classifications)

    @property
    def classifications(self) -> dict[str, dict[str, Any]]:
        """Return a defensive copy of the full classification mapping."""
        return dict(self._classifications)

    @classmethod
    def from_json_file(cls, path: Path) -> VerifiabilityClassifier:
        """Load classifier from a JSON file (see `alice_verifiability.json`)."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(classifications=data["classifications"])

    def is_public(self, rule: Rule) -> bool:
        """Return True iff the rule is PUBLIC. Raises KeyError if unknown."""
        entry = self._classifications[rule.id]  # KeyError if unknown
        return bool(entry["verifiability"] == "public")

    def partition_rules(
        self,
        rules: tuple[Rule, ...] | list[Rule],
    ) -> tuple[list[Rule], list[Rule]]:
        """Return (public_rules, private_rules). Skip unknown (pas de raise)."""
        pub: list[Rule] = []
        priv: list[Rule] = []
        for r in rules:
            if r.id not in self._classifications:
                continue
            if self.is_public(r):
                pub.append(r)
            else:
                priv.append(r)
        return pub, priv
