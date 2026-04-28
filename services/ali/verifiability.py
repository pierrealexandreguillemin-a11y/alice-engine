"""VerifiabilityClassifier — PUBLIC/PRIVATE/OUT_OF_SCOPE annotations ALICE.

ISO 24027 : fairness classification (distinguer ce qu'on peut vérifier
sur l'adversaire depuis ce qu'on doit supposer).
ISO 42001 : traceability des décisions de classification.
ISO 27034 : Pydantic input validation au load JSON (D-P3-07 résorption).

Key utilisé : `rule.id` (stable FFE label, ex. "N1-N4_3.7.a_001"),
PAS `rule.uuid` (code court chess-app).

Verifiability values (D-P3-09 résorption) :
- ``public`` : applicable au générateur Monte-Carlo (vérifiable sur adversaire).
- ``private`` : supposée respectée par l'adversaire (décisions internes club).
- ``out_of_scope`` : règle non-composition (ex arbitrage A02 §3.7).
  Distinct de ``private`` car n'a pas vocation à être supposée respectée
  par modèle composition adverse.

Document ID: ALICE-ALI-VERIFIABILITY
Version: 1.1.0
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

    from services.ffe.rule_engine import Rule


Verifiability = Literal["public", "private", "out_of_scope"]


class VerifiabilityEntry(BaseModel):
    """Pydantic schema for one rule classification entry (D-P3-07).

    Validates JSON load-time : missing fields, invalid verifiability values,
    extra unknown fields all raise ValidationError instead of latent KeyError.
    """

    verifiability: Verifiability = Field(
        ..., description="public | private | out_of_scope (D-P3-09)"
    )
    reason: str = Field(..., min_length=1, description="human-readable rationale")
    data_source: str = Field(default="-", description="parquet path or '-' if N/A")

    model_config = {"extra": "forbid"}


class VerifiabilityFile(BaseModel):
    """Pydantic schema for the full alice_verifiability.json structure."""

    metadata: dict[str, str] = Field(default_factory=dict)
    classifications: dict[str, VerifiabilityEntry] = Field(...)

    model_config = {"extra": "forbid"}


class VerifiabilityClassifier:
    """Classifie les règles FFE en PUBLIC / PRIVATE / OUT_OF_SCOPE.

    Key : `rule.id` (identifiant stable FFE article-based).
    """

    def __init__(self, classifications: dict[str, VerifiabilityEntry]) -> None:
        """Initialize classifier from validated entries."""
        self._classifications: dict[str, VerifiabilityEntry] = dict(classifications)

    @property
    def classifications(self) -> dict[str, VerifiabilityEntry]:
        """Return a defensive copy of the full classification mapping."""
        return dict(self._classifications)

    @classmethod
    def from_json_file(cls, path: Path) -> VerifiabilityClassifier:
        """Load + validate classifier (Pydantic, D-P3-07).

        @raises pydantic.ValidationError: malformed JSON (missing
                fields, invalid verifiability literal, extra keys).
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        validated = VerifiabilityFile.model_validate(raw)
        return cls(classifications=validated.classifications)

    def is_public(self, rule: Rule) -> bool:
        """Return True iff the rule is PUBLIC. Raises KeyError if unknown."""
        entry = self._classifications[rule.id]
        return entry.verifiability == "public"

    def partition_rules(
        self,
        rules: tuple[Rule, ...] | list[Rule],
    ) -> tuple[list[Rule], list[Rule]]:
        """Return (public_rules, private_rules).

        Both unknown rules and ``out_of_scope`` rules are skipped (neither
        enforced as PUBLIC by sampler nor supposed respected as PRIVATE).
        """
        pub: list[Rule] = []
        priv: list[Rule] = []
        for r in rules:
            entry = self._classifications.get(r.id)
            if entry is None:
                continue
            if entry.verifiability == "public":
                pub.append(r)
            elif entry.verifiability == "private":
                priv.append(r)
            # out_of_scope : skip both lists (D-P3-09)
        return pub, priv
