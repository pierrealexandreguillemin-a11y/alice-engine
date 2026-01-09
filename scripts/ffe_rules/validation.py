"""Validation de composition FFE - ISO 5055.

Ce module contient les fonctions de validation
de composition selon les regles FFE.
"""

from __future__ import annotations

from scripts.ffe_rules.brulage import calculer_pct_noyau, est_brule, valide_noyau
from scripts.ffe_rules.types import Equipe, Joueur, ReglesCompetition, Sexe


def _valider_ordre_elo(
    composition: list[Joueur],
    regles: ReglesCompetition,
) -> list[str]:
    """Valide l'ordre Elo de la composition."""
    erreurs: list[str] = []
    if regles.get("ordre_elo_obligatoire", True):
        for i in range(len(composition) - 1):
            if composition[i].elo < composition[i + 1].elo - 100:
                erreurs.append(f"Ordre Elo invalide: ech {i + 1}")
    return erreurs


def _valider_brulage(
    composition: list[Joueur],
    equipe: Equipe,
    historique_brulage: dict[int, dict[str, int]],
    regles: ReglesCompetition,
) -> list[str]:
    """Valide les joueurs brules."""
    erreurs: list[str] = []
    seuil_brulage = regles.get("seuil_brulage")
    if seuil_brulage is not None:
        for j in composition:
            if est_brule(j.id_fide, equipe.nom, historique_brulage, seuil_brulage):
                erreurs.append(f"{j.nom} est brule pour {equipe.nom}")
    return erreurs


def _valider_noyau_composition(
    composition: list[Joueur],
    equipe: Equipe,
    historique_noyau: dict[str, set[int]],
    regles: ReglesCompetition,
) -> list[str]:
    """Valide la regle du noyau."""
    erreurs: list[str] = []
    composition_ids = [j.id_fide for j in composition]
    if not valide_noyau(composition_ids, equipe, historique_noyau, regles):
        pct = calculer_pct_noyau(composition_ids, equipe.nom, historique_noyau)
        noyau_requis = regles.get("noyau", 0)
        erreurs.append(f"Noyau insuffisant: {pct:.0%} < {noyau_requis}%")
    return erreurs


def _valider_mutes(
    composition: list[Joueur],
    regles: ReglesCompetition,
) -> list[str]:
    """Valide le quota de mutes."""
    erreurs: list[str] = []
    max_mutes = regles.get("max_mutes")
    if max_mutes is not None:
        nb_mutes = sum(1 for j in composition if j.mute)
        if nb_mutes > max_mutes:
            erreurs.append(f"Trop de mutes: {nb_mutes} > {max_mutes}")
    return erreurs


def _valider_elo_max(
    composition: list[Joueur],
    equipe: Equipe,
    regles: ReglesCompetition,
) -> list[str]:
    """Valide l'Elo maximum."""
    erreurs: list[str] = []
    elo_max = regles.get("elo_max")
    if elo_max is None:
        return erreurs

    division_lower = equipe.division.lower()
    is_n4 = "n4" in division_lower or "nationale 4" in division_lower

    # Appliquer si: C03 (Loubatiere) ou N4 A02
    if is_n4 or elo_max < 2400:
        for j in composition:
            if j.elo > elo_max:
                erreurs.append(f"{j.nom} Elo {j.elo} > {elo_max}")
    return erreurs


def _valider_quota_nationalite(
    composition: list[Joueur],
    regles: ReglesCompetition,
) -> list[str]:
    """Valide le quota de nationalite."""
    erreurs: list[str] = []
    if not regles.get("quota_nationalite", False):
        return erreurs

    min_fr_eu = regles.get("min_fr_eu", 0)
    if min_fr_eu > 0:
        fr_eu = sum(1 for j in composition if j.nationalite in ("FRA", "FR", "UE"))
        if fr_eu < min_fr_eu:
            erreurs.append(f"Quota nationalite: {fr_eu} FR/UE < {min_fr_eu}")
    return erreurs


def _valider_elo_total(
    composition: list[Joueur],
    regles: ReglesCompetition,
) -> list[str]:
    """Valide l'Elo total maximum."""
    erreurs: list[str] = []
    elo_total_max = regles.get("elo_total_max")
    if elo_total_max is not None:
        elo_total = sum(j.elo for j in composition)
        if elo_total > elo_total_max:
            erreurs.append(f"Elo total {elo_total} > {elo_total_max}")
    return erreurs


def _valider_composition_obligatoire(
    composition: list[Joueur],
    regles: ReglesCompetition,
) -> list[str]:
    """Valide la composition obligatoire (mixte)."""
    erreurs: list[str] = []
    compo_oblig = regles.get("composition_obligatoire")
    if compo_oblig is None:
        return erreurs

    for sexe_key, nb_requis in compo_oblig.items():
        if sexe_key in ("hommes", "garcons"):
            nb = sum(1 for j in composition if j.sexe == Sexe.MASCULIN)
        else:
            nb = sum(1 for j in composition if j.sexe == Sexe.FEMININ)
        if nb < nb_requis:
            erreurs.append(f"{sexe_key}: {nb} < {nb_requis}")
    return erreurs


def valider_composition(
    composition: list[Joueur],
    equipe: Equipe,
    historique_brulage: dict[int, dict[str, int]],
    historique_noyau: dict[str, set[int]],
    regles: ReglesCompetition,
) -> list[str]:
    """Valide une composition selon les regles FFE.

    Args:
    ----
        composition: Liste des joueurs alignes
        equipe: Equipe concernee
        historique_brulage: {joueur_id: {equipe_nom: nb_matchs}}
        historique_noyau: {equipe_nom: set(joueur_ids)}
        regles: Regles applicables a cette competition

    Returns:
    -------
        Liste des erreurs detectees (vide si valide)
    """
    if not composition:
        return ["Composition vide"]

    erreurs: list[str] = []

    # Validations modulaires
    erreurs.extend(_valider_ordre_elo(composition, regles))
    erreurs.extend(_valider_brulage(composition, equipe, historique_brulage, regles))
    erreurs.extend(_valider_noyau_composition(composition, equipe, historique_noyau, regles))
    erreurs.extend(_valider_mutes(composition, regles))
    erreurs.extend(_valider_elo_max(composition, equipe, regles))
    erreurs.extend(_valider_quota_nationalite(composition, regles))
    erreurs.extend(_valider_elo_total(composition, regles))
    erreurs.extend(_valider_composition_obligatoire(composition, regles))

    return erreurs
