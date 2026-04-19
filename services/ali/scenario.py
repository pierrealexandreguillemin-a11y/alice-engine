"""ALI scenario types — Plan 2 Generator SOTA.

ISO 5055 : SRP, frozen dataclasses.
ISO 29119 : immutable, comparable par valeur.
ISO 5259 : lineage_hash propage dans ScenarioSet.

Document ID: ALICE-ALI-SCENARIO-TYPES
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from services.ali.types import PlayerCandidate

_EXPECTED_SCENARIOS = 20
_WEIGHTS_SUM_TOLERANCE = 1e-4


@dataclass(frozen=True)
class BoardAssignment:
    """Un joueur assigne a un echiquier dans un lineup."""

    board: int
    player: PlayerCandidate
    p_assignment: float  # P(ce joueur a cet echiquier | present)


@dataclass(frozen=True)
class Lineup:
    """Composition complete d'une equipe (team_size joueurs)."""

    team_size: int
    assignments: tuple[BoardAssignment, ...]


@dataclass(frozen=True)
class Scenario:
    """Un scenario (lineup + poids dans ScenarioSet)."""

    lineup: Lineup
    joint_prob: float  # unnormalized
    weight: float  # normalized (sum = 1 over set)
    source: Literal["topk", "monte_carlo"]


@dataclass(frozen=True)
class ScenarioSet:
    """20 scenarios ponderes ALI pour un match adverse."""

    scenarios: tuple[Scenario, ...]
    opponent_club_id: str
    round_date: str  # YYYY-MM-DD
    generated_at: str  # ISO 8601 UTC
    lineage_hash: str  # SHA-256 hex (64 chars)

    def validate(self) -> None:
        """Validate ScenarioSet structurally (T18 / T19 quality gates).

        Raises ValueError on violation.

        T20 (uniqueness par signature) n'est PAS verifie ici : c'est
        au generator (Task 7) d'en garantir l'invariant. `validate()`
        reste une verification structurelle pure (count + weights sum)
        pour permettre des sets degenerate en tests unitaires.
        """
        if len(self.scenarios) != _EXPECTED_SCENARIOS:
            raise ValueError(
                f"ScenarioSet must contain {_EXPECTED_SCENARIOS} scenarios, "
                f"got {len(self.scenarios)}",
            )
        weights_sum = sum(s.weight for s in self.scenarios)
        if abs(weights_sum - 1.0) > _WEIGHTS_SUM_TOLERANCE:
            raise ValueError(
                f"weights sum {weights_sum} != 1.0 " f"(tolerance {_WEIGHTS_SUM_TOLERANCE})",
            )
