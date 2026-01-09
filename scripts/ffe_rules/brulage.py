"""Regles de brulage et noyau FFE - ISO 5055.

Ce module contient les regles de brulage de joueurs
et de noyau d'equipe.
"""

from __future__ import annotations

from scripts.ffe_rules.competition import get_niveau_equipe
from scripts.ffe_rules.types import Equipe, ReglesCompetition


def est_brule(
    joueur_id: int | str,
    equipe_cible: str,
    historique: dict[int | str, dict[str, int]],
    seuil_brulage: int = 3,
) -> bool:
    """Verifie si un joueur est brule pour une equipe.

    Un joueur est brule s'il a joue `seuil_brulage` fois ou plus
    dans une equipe de niveau superieur.

    Args:
    ----
        joueur_id: ID FIDE du joueur
        equipe_cible: Nom de l'equipe cible
        historique: {joueur_id: {equipe_nom: nb_matchs}}
        seuil_brulage: Nombre de matchs avant brulage (3 pour A02)

    Returns:
    -------
        True si le joueur est brule pour cette equipe
    """
    joueur_hist = historique.get(joueur_id, {})
    niveau_cible = get_niveau_equipe(equipe_cible)

    for eq_nom, nb_matchs in joueur_hist.items():
        niveau_eq = get_niveau_equipe(eq_nom)
        if niveau_eq < niveau_cible and nb_matchs >= seuil_brulage:
            return True
    return False


def matchs_avant_brulage(
    joueur_id: int | str,
    equipe_superieure: str,
    historique: dict[int | str, dict[str, int]],
    seuil_brulage: int = 3,
) -> int:
    """Calcule le nombre de matchs restants avant brulage.

    Args:
    ----
        joueur_id: ID FIDE du joueur
        equipe_superieure: Equipe de niveau superieur
        historique: {joueur_id: {equipe_nom: nb_matchs}}
        seuil_brulage: Nombre de matchs avant brulage

    Returns:
    -------
        Nombre de matchs encore possibles (0 = deja brule)
    """
    joueur_hist = historique.get(joueur_id, {})
    matchs_joues = joueur_hist.get(equipe_superieure, 0)
    return max(0, seuil_brulage - matchs_joues)


def get_noyau(
    equipe_nom: str,
    historique_noyau: dict[str, set[int | str]],
) -> set[int | str]:
    """Retourne l'ensemble des joueurs du noyau d'une equipe.

    Args:
    ----
        equipe_nom: Nom de l'equipe
        historique_noyau: {equipe_nom: set(joueur_ids)}

    Returns:
    -------
        Set des IDs joueurs ayant deja joue pour cette equipe
    """
    return historique_noyau.get(equipe_nom, set())


def calculer_pct_noyau(
    composition_ids: list[int | str],
    equipe_nom: str,
    historique_noyau: dict[str, set[int | str]],
) -> float:
    """Calcule le pourcentage de joueurs du noyau dans une composition.

    Args:
    ----
        composition_ids: Liste des IDs joueurs de la composition
        equipe_nom: Nom de l'equipe
        historique_noyau: {equipe_nom: set(joueur_ids)}

    Returns:
    -------
        Pourcentage (0.0 - 1.0) de joueurs du noyau
    """
    if not composition_ids:
        return 0.0

    noyau_ids = get_noyau(equipe_nom, historique_noyau)
    nb_noyau = sum(1 for jid in composition_ids if jid in noyau_ids)
    return nb_noyau / len(composition_ids)


def valide_noyau(
    composition_ids: list[int | str],
    equipe: Equipe,
    historique_noyau: dict[str, set[int | str]],
    regles: ReglesCompetition,
) -> bool:
    """Verifie si la composition respecte la regle du noyau.

    Args:
    ----
        composition_ids: Liste des IDs joueurs
        equipe: Equipe concernee
        historique_noyau: {equipe_nom: set(joueur_ids)}
        regles: Regles applicables

    Returns:
    -------
        True si le noyau est respecte
    """
    noyau_requis = regles.get("noyau")
    if noyau_requis is None or equipe.ronde <= 1:
        return True  # Pas de regle ou ronde 1

    noyau_type = regles.get("noyau_type", "pourcentage")
    noyau_ids = get_noyau(equipe.nom, historique_noyau)
    nb_noyau = sum(1 for jid in composition_ids if jid in noyau_ids)

    if noyau_type == "pourcentage":
        pct = nb_noyau / len(composition_ids) if composition_ids else 0
        return pct >= noyau_requis / 100
    return nb_noyau >= noyau_requis
