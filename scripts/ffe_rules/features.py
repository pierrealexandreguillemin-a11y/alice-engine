"""Calcul de features reglementaires FFE - ISO 5055.

Ce module contient les fonctions de calcul de features
pour les modeles ML.
"""

from __future__ import annotations

from scripts.ffe_rules.brulage import est_brule, get_noyau
from scripts.ffe_rules.competition import get_niveau_equipe
from scripts.ffe_rules.types import FeaturesReglementaires, ReglesCompetition


def calculer_features_joueur(
    joueur_id: int | str,
    equipe_nom: str,
    ronde: int,
    historique_brulage: dict[int | str, dict[str, int]],
    historique_noyau: dict[str, set[int | str]],
    historique_parties: dict[int | str, int],
    regles: ReglesCompetition,
) -> FeaturesReglementaires:
    """Calcule les features reglementaires pour un joueur.

    Args:
    ----
        joueur_id: ID FIDE du joueur
        equipe_nom: Nom de l'equipe
        ronde: Numero de la ronde
        historique_brulage: {joueur_id: {equipe_nom: nb_matchs}}
        historique_noyau: {equipe_nom: set(joueur_ids)}
        historique_parties: {joueur_id: nb_parties_saison}
        regles: Regles de la competition

    Returns:
    -------
        FeaturesReglementaires avec toutes les features calculees
    """
    seuil = regles.get("seuil_brulage", 3)
    max_parties = regles.get("max_parties_saison")

    # Brulage
    brule = est_brule(joueur_id, equipe_nom, historique_brulage, seuil) if seuil else False

    # Matchs restants
    matchs_restants = _calculer_matchs_restants(joueur_id, equipe_nom, historique_brulage, seuil)

    # Parties jouees
    nb_parties = historique_parties.get(joueur_id, 0)
    peut_jouer = _peut_jouer_ronde(nb_parties, ronde, max_parties)

    # Noyau
    est_noyau = joueur_id in get_noyau(equipe_nom, historique_noyau)

    return FeaturesReglementaires(
        joueur_brule=brule,
        matchs_avant_brulage=matchs_restants,
        nb_matchs_joues_saison=nb_parties,
        peut_jouer_ronde_n=peut_jouer,
        est_dans_noyau=est_noyau,
    )


def _calculer_matchs_restants(
    joueur_id: int | str,
    equipe_nom: str,
    historique_brulage: dict[int | str, dict[str, int]],
    seuil: int | None,
) -> int:
    """Calcule le nombre de matchs restants avant brulage."""
    if not seuil:
        return 0

    joueur_hist = historique_brulage.get(joueur_id, {})
    niveau_cible = get_niveau_equipe(equipe_nom)
    max_matchs_sup = 0

    for eq_nom, nb_matchs in joueur_hist.items():
        if get_niveau_equipe(eq_nom) < niveau_cible:
            max_matchs_sup = max(max_matchs_sup, nb_matchs)

    return max(0, seuil - max_matchs_sup)


def _peut_jouer_ronde(
    nb_parties: int,
    ronde: int,
    max_parties: int | None,
) -> bool:
    """Determine si le joueur peut jouer a cette ronde."""
    if max_parties is not None:
        return nb_parties < max_parties
    return nb_parties < ronde
