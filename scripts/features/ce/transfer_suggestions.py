"""Module: transfer_suggestions.py - Suggestions de Transferts.

Extrait de transferability.py pour conformité ISO 5055 (<300 lignes).

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)
- ISO/IEC 5259:2024 - Data Quality

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.features.ce.types import TeamTransferability

if TYPE_CHECKING:
    import pandas as pd


def calculate_transfer_score(team: TeamTransferability) -> float:
    """Calcule un score de transfert [-1, 1].

    -1.0 = Donneur maximal (équipe condamnée)
    +1.0 = Receveur prioritaire (course titre urgente)
     0.0 = Neutre

    Ce score permet de matcher donneurs et receveurs.
    """
    if team.can_donate and not team.can_receive:
        return -1.0 + (team.priority - 1) * 0.2

    if team.can_receive and not team.can_donate:
        return 1.0 - (team.priority - 1) * 0.2

    return 0.0


def suggest_transfers(
    transferability: pd.DataFrame,
    club_id: str | None = None,
) -> list[dict]:
    """Suggère des transferts de joueurs entre équipes.

    Args:
    ----
        transferability: DataFrame issu de calculate_transferability
        club_id: Optionnel - filtrer par club

    Returns:
    -------
        Liste de suggestions:
        [{"from_team": ..., "to_team": ..., "priority": ..., "reason": ...}]
    """
    if transferability.empty:
        return []

    donors = transferability[transferability["can_donate"]].sort_values("priority")
    receivers = transferability[transferability["can_receive"]].sort_values("priority")

    suggestions = []

    for _, donor in donors.iterrows():
        for _, receiver in receivers.iterrows():
            if donor["saison"] != receiver["saison"]:
                continue

            suggestions.append(
                {
                    "from_team": donor["equipe"],
                    "to_team": receiver["equipe"],
                    "donor_scenario": donor["scenario"],
                    "receiver_scenario": receiver["scenario"],
                    "priority": receiver["priority"],
                    "reason": f"{donor['reason']} → {receiver['reason']}",
                }
            )

    return sorted(suggestions, key=lambda x: x["priority"])
