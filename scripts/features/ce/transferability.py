"""Module: scripts/features/ce/transferability.py - Transférabilité Joueurs.

Document ID: ALICE-MOD-TRANSFER-001
Version: 1.0.0

Ce module détermine si les joueurs d'une équipe peuvent être
"prêtés" à une autre équipe du club pour renforcer celle-ci.

Cas d'usage (DOCUMENTÉS ISO 5259):
- Équipe condamnée (relégation mathématique): peut être affaiblie
- Équipe promue (montée assurée): peut être affaiblie
- Équipe en course titre: prioritaire pour recevoir des renforts
- Équipe en danger: peut recevoir des renforts

Règle FFE (DOCUMENTÉE):
- Un joueur ne peut jouer que dans UNE équipe par ronde
- Mais peut changer d'équipe entre les rondes
- Le classement Elo détermine l'équipe "naturelle" (hiérarchie)

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (<300 lignes, responsabilité unique)
- ISO/IEC 5259:2024 - Data Quality (features depuis données réelles)
- ISO/IEC 42001:2023 - AI Management (traçabilité décisions)

See Also
--------
- scripts/features/ce/scenarios.py - Scénarios tactiques
- scripts/features/ce/urgency.py - Calcul urgence
- tests/test_features_ce.py - Tests unitaires

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TeamTransferability:
    """État de transférabilité d'une équipe."""

    equipe: str
    saison: int
    ronde: int
    scenario: str
    can_donate: bool  # Peut céder des joueurs
    can_receive: bool  # Peut recevoir des renforts
    priority: int  # Priorité (1 = max, 5 = min)
    reason: str  # Justification


def calculate_transferability(
    scenarios: pd.DataFrame,
    urgency: pd.DataFrame,
    ronde_actuelle: int,
    nb_rondes_total: int = 9,
) -> pd.DataFrame:
    """Calcule la transférabilité des joueurs entre équipes du club.

    Args:
    ----
        scenarios: DataFrame avec colonnes scenario, urgence_score
        urgency: DataFrame avec colonnes montee_possible, maintien_assure
        ronde_actuelle: Numéro de la ronde actuelle
        nb_rondes_total: Nombre total de rondes

    Returns:
    -------
        DataFrame avec:
        - equipe: nom équipe
        - can_donate: bool - équipe peut céder des joueurs
        - can_receive: bool - équipe peut recevoir des renforts
        - priority: int - priorité de renforcement (1=max)
        - transfer_score: float [-1, 1] - négatif=donneur, positif=receveur
        - reason: str - justification

    ISO 5259: Logique de transfert basée sur situation mathématique.
    """
    logger.info("Calcul transférabilité inter-équipes...")

    if scenarios.empty or urgency.empty:
        return pd.DataFrame()

    rondes_restantes = max(0, nb_rondes_total - ronde_actuelle)

    # Fusionner scenarios et urgency
    merged = pd.merge(
        scenarios,
        urgency[
            ["equipe", "saison", "ronde", "montee_possible", "maintien_assure", "points_cumules"]
        ],
        on=["equipe", "saison", "ronde"],
        how="left",
    )

    transferability_data = []

    for _, row in merged.iterrows():
        result = _evaluate_team_transferability(
            equipe=row["equipe"],
            saison=row.get("saison", 2025),
            ronde=row.get("ronde", ronde_actuelle),
            scenario=row["scenario"],
            urgence_score=row["urgence_score"],
            montee_possible=row.get("montee_possible", True),
            maintien_assure=row.get("maintien_assure", False),
            position=row.get("position", 1),
            points_cumules=row.get("points_cumules", 0),
            rondes_restantes=rondes_restantes,
        )

        transferability_data.append(
            {
                "equipe": result.equipe,
                "saison": result.saison,
                "ronde": result.ronde,
                "scenario": result.scenario,
                "can_donate": result.can_donate,
                "can_receive": result.can_receive,
                "priority": result.priority,
                "transfer_score": _calculate_transfer_score(result),
                "reason": result.reason,
            }
        )

    result_df = pd.DataFrame(transferability_data)
    logger.info(f"  {len(result_df)} équipes avec transférabilité")
    return result_df


def _evaluate_team_transferability(  # noqa: PLR0911 - Multiple returns for clarity
    equipe: str,
    saison: int,
    ronde: int,
    scenario: str,
    urgence_score: float,
    montee_possible: bool,
    maintien_assure: bool,
    position: int,
    points_cumules: int,
    rondes_restantes: int,
) -> TeamTransferability:
    """Évalue la transférabilité d'une équipe.

    Logique métier (ISO 5259 - documentée):
    1. Équipe condamnée (scenario='condamne') → DONNEUR
    2. Équipe promue (maintien_assure + position <= 2) → DONNEUR potentiel
    3. Équipe en course titre (scenario='course_titre') → RECEVEUR prioritaire
    4. Équipe en danger (scenario='danger') → RECEVEUR secondaire
    5. Mi-tableau → Neutre
    """
    # Cas 1: Équipe mathématiquement condamnée
    if scenario == "condamne":
        return TeamTransferability(
            equipe=equipe,
            saison=saison,
            ronde=ronde,
            scenario=scenario,
            can_donate=True,
            can_receive=False,
            priority=5,  # Dernière priorité
            reason="Relégation mathématique - joueurs disponibles pour renfort",
        )

    # Cas 2: Course au titre avec urgence haute (PRIORITAIRE)
    if scenario == "course_titre" and urgence_score >= 0.8:
        return TeamTransferability(
            equipe=equipe,
            saison=saison,
            ronde=ronde,
            scenario=scenario,
            can_donate=False,
            can_receive=True,
            priority=1,  # Priorité maximale
            reason="Course au titre urgente - prioritaire pour renforts",
        )

    # Cas 3: Course au titre normale
    if scenario == "course_titre":
        return TeamTransferability(
            equipe=equipe,
            saison=saison,
            ronde=ronde,
            scenario=scenario,
            can_donate=False,
            can_receive=True,
            priority=2,
            reason="Course au titre - peut recevoir renforts",
        )

    # Cas 4: Équipe en danger
    if scenario == "danger":
        return TeamTransferability(
            equipe=equipe,
            saison=saison,
            ronde=ronde,
            scenario=scenario,
            can_donate=False,
            can_receive=True,
            priority=2,
            reason="Danger relégation - peut recevoir renforts",
        )

    # Cas 5: Course montée
    if scenario == "course_montee":
        return TeamTransferability(
            equipe=equipe,
            saison=saison,
            ronde=ronde,
            scenario=scenario,
            can_donate=False,
            can_receive=True,
            priority=3,
            reason="Course montée - peut recevoir renforts si dispo",
        )

    # Cas 6: Montée/maintien assuré en dernières rondes (mi-tableau avec objectif atteint)
    # SEULEMENT pour les équipes qui ne sont PAS en course (mi_tableau)
    if maintien_assure and position <= 2 and rondes_restantes <= 1:
        return TeamTransferability(
            equipe=equipe,
            saison=saison,
            ronde=ronde,
            scenario=scenario,
            can_donate=True,
            can_receive=False,
            priority=4,
            reason="Objectif atteint - joueurs peuvent renforcer autre équipe",
        )

    # Cas 7: Mi-tableau - neutre
    return TeamTransferability(
        equipe=equipe,
        saison=saison,
        ronde=ronde,
        scenario=scenario,
        can_donate=False,
        can_receive=False,
        priority=4,
        reason="Mi-tableau - pas de transfert nécessaire",
    )


def _calculate_transfer_score(team: TeamTransferability) -> float:
    """Calcule un score de transfert [-1, 1].

    -1.0 = Donneur maximal (équipe condamnée)
    +1.0 = Receveur prioritaire (course titre urgente)
     0.0 = Neutre

    Ce score permet de matcher donneurs et receveurs.
    """
    if team.can_donate and not team.can_receive:
        # Donneur: score négatif selon priorité
        return -1.0 + (team.priority - 1) * 0.2  # -1.0 à -0.2

    if team.can_receive and not team.can_donate:
        # Receveur: score positif selon priorité
        return 1.0 - (team.priority - 1) * 0.2  # +1.0 à +0.2

    # Neutre
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

    # Trier par priorité du receveur
    return sorted(suggestions, key=lambda x: x["priority"])
