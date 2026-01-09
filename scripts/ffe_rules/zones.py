"""Zones d'enjeu et mouvements joueurs - ISO 5055.

Ce module contient les calculs de zones d'enjeu
et la detection de mouvements de joueurs.
"""

from __future__ import annotations

from scripts.ffe_rules.competition import get_niveau_equipe
from scripts.ffe_rules.types import MouvementJoueur


def calculer_zone_enjeu(
    position: int,
    nb_equipes: int,
    division: str,
) -> str:
    """Determine la zone d'enjeu d'une equipe.

    Args:
    ----
        position: Position actuelle au classement (1-based)
        nb_equipes: Nombre total d'equipes dans le groupe
        division: Nom de la division

    Returns:
    -------
        Zone d'enjeu: "montee" | "course_titre" | "mi_tableau" | "danger" | "descente"
    """
    if position == 1:
        return "montee"

    division_lower = division.lower()

    # Check descente based on division
    if _is_in_descente_zone(position, division_lower):
        return "descente"

    # Intermediate zones
    if position <= 3:
        return "course_titre"
    if division_lower == "n3" and position >= 6:
        return "danger"
    if position >= nb_equipes - 3:
        return "danger"

    return "mi_tableau"


def _is_in_descente_zone(position: int, division_lower: str) -> bool:
    """Check if position is in relegation zone for division."""
    if "top16" in division_lower or "top 16" in division_lower:
        return position >= 13
    if division_lower in ("n1", "n2"):
        return position >= 9
    if division_lower == "n3":
        return position >= 8
    return False


def calculer_ecart_objectif(
    points_equipe: int,
    points_objectif: int,
) -> int:
    """Calcule l'ecart par rapport a l'objectif (montee ou maintien).

    Args:
    ----
        points_equipe: Points actuels de l'equipe
        points_objectif: Points de l'objectif (1er pour montee, dernier non releguable)

    Returns:
    -------
        Ecart en points (positif = en avance, negatif = en retard)
    """
    return points_equipe - points_objectif


def detecter_mouvement_joueur(
    joueur_elo: int,
    equipe_avant: str,
    equipe_apres: str,
) -> MouvementJoueur:
    """Detecte et categorise un mouvement de joueur entre equipes.

    Args:
    ----
        joueur_elo: Elo du joueur
        equipe_avant: Nom equipe d'origine
        equipe_apres: Nom equipe destination

    Returns:
    -------
        MouvementJoueur avec type, equipes impactees et delta elo
    """
    niveau_avant = get_niveau_equipe(equipe_avant)
    niveau_apres = get_niveau_equipe(equipe_apres)

    if niveau_apres < niveau_avant:  # Monte (N2 -> N1)
        return MouvementJoueur(
            type="promotion",
            equipe_renforcee=equipe_apres,
            equipe_affaiblie=equipe_avant,
            impact=joueur_elo,
        )
    if niveau_apres > niveau_avant:  # Descend (N1 -> N2)
        return MouvementJoueur(
            type="relegation",
            equipe_renforcee=equipe_apres,
            equipe_affaiblie=equipe_avant,
            impact=joueur_elo,
        )
    return MouvementJoueur(
        type="lateral",
        equipe_renforcee=None,
        equipe_affaiblie=None,
        impact=0,
    )
