#!/usr/bin/env python3
"""
Features reglementaires FFE pour ALICE.

Ce module implemente les regles FFE sous forme de features exploitables
par le modele ML ALI (Adversarial Lineup Inference).

Regles implementees:
- Joueur brule (A02 Art. 3.7.c)
- Noyau 50% (A02 Art. 3.7.f)
- Quotas mutes (A02 Art. 3.7.g)
- Detection type competition
- Zones d'enjeu classement

Conformite ISO/IEC 5055 (typage strict), ISO 25010 (exactitude fonctionnelle).

Usage:
    from scripts.ffe_rules_features import (
        TypeCompetition,
        detecter_type_competition,
        get_regles_competition,
        calculer_features_reglementaires,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict

# ==============================================================================
# TYPES DE DONNEES (ISO/IEC 5055 - Typage strict)
# ==============================================================================


class TypeCompetition(Enum):
    """Types de competition FFE."""

    A02 = "A02"  # CFC Hommes
    F01 = "F01"  # CFC Feminin
    C01 = "C01"  # Coupe de France
    C03 = "C03"  # Coupe Loubatiere
    C04 = "C04"  # Coupe Parite
    J02 = "J02"  # Interclubs Jeunes
    J03 = "J03"  # Scolaire
    REG = "REG"  # Regionale
    DEP = "DEP"  # Departemental


class NiveauCompetition(Enum):
    """Niveaux hierarchiques."""

    TOP16 = "top16"
    N1 = "n1"
    N2 = "n2"
    N3 = "n3"
    N4 = "n4"
    REGIONAL = "regional"
    DEPARTEMENTAL = "departemental"
    COUPE = "coupe"
    INCONNU = "inconnu"


class Sexe(Enum):
    """Sexe du joueur."""

    MASCULIN = "M"
    FEMININ = "F"


class ReglesCompetition(TypedDict, total=False):
    """Structure typee des regles par competition."""

    taille_equipe: int | dict[str, int]
    seuil_brulage: int | None
    max_parties_saison: int | None
    max_mutes: int | None
    min_fr_eu: int | None
    ordre_elo_obligatoire: bool
    elo_max: int | None
    elo_total_max: int | None
    noyau: int | None
    noyau_type: str  # "pourcentage" | "absolu"
    quota_nationalite: bool
    composition_obligatoire: dict[str, int] | None
    categories_age: dict[int, str] | None


class MouvementJoueur(TypedDict):
    """Resultat detection mouvement."""

    type: str  # "promotion" | "relegation" | "lateral"
    equipe_renforcee: str | None
    equipe_affaiblie: str | None
    impact: int


class FeaturesReglementaires(TypedDict, total=False):
    """Features calculees pour un joueur dans une composition."""

    joueur_brule: bool
    matchs_avant_brulage: int  # 0-3 (ou seuil selon competition)
    nb_matchs_joues_saison: int
    peut_jouer_ronde_n: bool
    est_dans_noyau: bool
    pct_noyau_equipe: float
    joueur_mute: bool
    nb_mutes_deja_alignes: int
    zone_enjeu_equipe: str  # "montee" | "danger" | "mi_tableau" | "course_titre"
    ecart_objectif: int


@dataclass(frozen=True)
class Joueur:
    """Representation d'un joueur."""

    id_fide: int
    nom: str
    elo: int
    sexe: Sexe
    nationalite: str
    mute: bool
    date_naissance: str | None = None


@dataclass
class Equipe:
    """Representation d'une equipe."""

    nom: str
    club: str
    division: str
    ronde: int
    groupe: str | None = None


@dataclass
class HistoriqueJoueur:
    """Historique de participation d'un joueur."""

    joueur_id: int
    matchs_par_equipe: dict[str, int] = field(default_factory=dict)
    matchs_total_saison: int = 0
    equipes_jouees: set[str] = field(default_factory=set)


# ==============================================================================
# DETECTION TYPE COMPETITION
# ==============================================================================


def detecter_type_competition(nom_competition: str) -> TypeCompetition:
    """
    Detecte le type de competition pour appliquer les bonnes regles.

    Args:
        nom_competition: Nom de la competition (ex: "Nationale 2", "Coupe Loubatiere")

    Returns:
        TypeCompetition enum correspondant
    """
    nom = nom_competition.lower()

    # Coupes speciales (prioritaire - detecter avant "coupe france")
    if "loubatiere" in nom or "loubatière" in nom:
        return TypeCompetition.C03
    if "parite" in nom or "parité" in nom:
        return TypeCompetition.C04

    # Feminin
    if "feminin" in nom or "féminin" in nom or "12f" in nom or "feminine" in nom:
        return TypeCompetition.F01

    # Coupe de France (apres coupes speciales)
    if "coupe" in nom and "france" in nom:
        return TypeCompetition.C01

    # Scolaire (avant jeunes general)
    if "scolaire" in nom or "etablissement" in nom or "école" in nom:
        return TypeCompetition.J03

    # Jeunes
    if "jeune" in nom or "junior" in nom or "top jeunes" in nom:
        return TypeCompetition.J02

    # Regional / Departemental
    if any(x in nom for x in ["regionale", "régionale", "r1 ", "r2 ", "r3 "]):
        return TypeCompetition.REG
    if any(x in nom for x in ["departemental", "départemental", "n5", "n6", "criterium"]):
        return TypeCompetition.DEP

    # National (default pour Top16-N4)
    if any(x in nom for x in ["top 16", "n1", "n2", "n3", "n4", "nationale"]):
        return TypeCompetition.A02

    return TypeCompetition.A02  # Default


def get_niveau_equipe(equipe: str) -> int:
    """
    Retourne le niveau hierarchique d'une equipe (1=Top16, 8=N4).

    Args:
        equipe: Nom de l'equipe incluant sa division

    Returns:
        int: Niveau (1=plus fort, 8=plus faible)
    """
    equipe_lower = equipe.lower()
    niveaux: dict[str, int] = {
        # Hommes
        "top 16": 1,
        "top16": 1,
        "n1": 2,
        "nationale 1": 2,
        "n2": 3,
        "nationale 2": 3,
        "n3": 4,
        "nationale 3": 4,
        "n4": 5,
        "nationale 4": 5,
        # Feminin
        "top12f": 1,
        "top 12f": 1,
        "n1f": 2,
        "n2f": 3,
        # Regional/Departemental
        "regionale": 6,
        "r1": 6,
        "r2": 7,
        "r3": 8,
        "departemental": 9,
        "n5": 9,
        "n6": 10,
    }
    for pattern, niveau in niveaux.items():
        if pattern in equipe_lower:
            return niveau
    return 10  # Inconnu = plus faible


# ==============================================================================
# REGLES PAR COMPETITION
# ==============================================================================


def get_regles_a02() -> ReglesCompetition:
    """Regles du Championnat de France des Clubs (A02)."""
    return ReglesCompetition(
        taille_equipe=8,
        seuil_brulage=3,
        max_parties_saison=None,  # < numero_ronde
        max_mutes=3,
        min_fr_eu=5,
        ordre_elo_obligatoire=True,
        elo_max=2400,  # N4 uniquement
        noyau=50,
        noyau_type="pourcentage",
        quota_nationalite=True,
    )


def get_regles_feminin() -> ReglesCompetition:
    """Regles du Championnat Feminin (F01)."""
    return ReglesCompetition(
        taille_equipe=4,
        seuil_brulage=1,  # 1 match en Top12F = brule
        max_parties_saison=7,
        max_mutes=2,
        min_fr_eu=3,
        ordre_elo_obligatoire=True,
        quota_nationalite=True,
    )


def get_regles_coupe() -> ReglesCompetition:
    """Regles de la Coupe de France (C01)."""
    return ReglesCompetition(
        taille_equipe=4,
        seuil_brulage=None,
        max_mutes=2,
        min_fr_eu=2,
        ordre_elo_obligatoire=False,  # Capitaine libre!
        quota_nationalite=True,
    )


def get_regles_loubatiere() -> ReglesCompetition:
    """Regles de la Coupe Loubatiere (C03) - bas Elo."""
    return ReglesCompetition(
        taille_equipe=4,
        elo_max=1800,  # Strict!
        ordre_elo_obligatoire=False,
        min_fr_eu=3,
        max_mutes=None,
    )


def get_regles_parite() -> ReglesCompetition:
    """Regles de la Coupe de la Parite (C04) - mixte."""
    return ReglesCompetition(
        taille_equipe=4,
        elo_total_max=8000,
        ordre_elo_obligatoire=True,
        min_fr_eu=3,
        composition_obligatoire={"hommes": 2, "femmes": 2},
    )


def get_regles_jeunes() -> ReglesCompetition:
    """Regles des Interclubs Jeunes (J02)."""
    return ReglesCompetition(
        taille_equipe={"top": 8, "n1": 8, "n2": 8, "n3": 4},
        seuil_brulage=4,
        max_parties_saison=11,
        max_mutes=2,
        ordre_elo_obligatoire=False,  # Ordre par age
        categories_age={
            1: "U16",
            2: "U16",
            3: "U14",
            4: "U14",
            5: "U12",
            6: "U12",
            7: "U10",
            8: "U10",
        },
    )


def get_regles_scolaire() -> ReglesCompetition:
    """Regles du Championnat Scolaire (J03)."""
    return ReglesCompetition(
        taille_equipe=8,
        ordre_elo_obligatoire=False,  # Elo FIDE d'abord
        composition_obligatoire={"garcons": 2, "filles": 2},
    )


def get_regles_regionale() -> ReglesCompetition:
    """Regles des championnats regionaux."""
    return ReglesCompetition(
        taille_equipe=5,
        noyau=2,
        noyau_type="absolu",
        quota_nationalite=False,
    )


def get_regles_departemental() -> ReglesCompetition:
    """Regles des championnats departementaux."""
    return ReglesCompetition(
        taille_equipe=4,
        quota_nationalite=False,
    )


def get_regles_competition(type_comp: TypeCompetition) -> ReglesCompetition:
    """
    Retourne les regles applicables selon le type de competition.

    Args:
        type_comp: Type de competition (enum TypeCompetition)

    Returns:
        ReglesCompetition avec toutes les regles applicables
    """
    REGLES: dict[TypeCompetition, ReglesCompetition] = {
        TypeCompetition.A02: get_regles_a02(),
        TypeCompetition.F01: get_regles_feminin(),
        TypeCompetition.C01: get_regles_coupe(),
        TypeCompetition.C03: get_regles_loubatiere(),
        TypeCompetition.C04: get_regles_parite(),
        TypeCompetition.J02: get_regles_jeunes(),
        TypeCompetition.J03: get_regles_scolaire(),
        TypeCompetition.REG: get_regles_regionale(),
        TypeCompetition.DEP: get_regles_departemental(),
    }
    return REGLES.get(type_comp, get_regles_a02())


# ==============================================================================
# REGLES JOUEUR BRULE
# ==============================================================================


def est_brule(
    joueur_id: int | str,
    equipe_cible: str,
    historique: dict[int | str, dict[str, int]],
    seuil_brulage: int = 3,
) -> bool:
    """
    Verifie si un joueur est brule pour une equipe.

    Un joueur est brule s'il a joue `seuil_brulage` fois ou plus
    dans une equipe de niveau superieur.

    Args:
        joueur_id: ID FIDE du joueur
        equipe_cible: Nom de l'equipe cible
        historique: {joueur_id: {equipe_nom: nb_matchs}}
        seuil_brulage: Nombre de matchs avant brulage (3 pour A02)

    Returns:
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
    """
    Calcule le nombre de matchs restants avant brulage.

    Args:
        joueur_id: ID FIDE du joueur
        equipe_superieure: Equipe de niveau superieur
        historique: {joueur_id: {equipe_nom: nb_matchs}}
        seuil_brulage: Nombre de matchs avant brulage

    Returns:
        Nombre de matchs encore possibles (0 = deja brule)
    """
    joueur_hist = historique.get(joueur_id, {})
    matchs_joues = joueur_hist.get(equipe_superieure, 0)
    return max(0, seuil_brulage - matchs_joues)


# ==============================================================================
# REGLES NOYAU
# ==============================================================================


def get_noyau(
    equipe_nom: str,
    historique_noyau: dict[str, set[int | str]],
) -> set[int | str]:
    """
    Retourne l'ensemble des joueurs du noyau d'une equipe.

    Args:
        equipe_nom: Nom de l'equipe
        historique_noyau: {equipe_nom: set(joueur_ids)}

    Returns:
        Set des IDs joueurs ayant deja joue pour cette equipe
    """
    return historique_noyau.get(equipe_nom, set())


def calculer_pct_noyau(
    composition_ids: list[int | str],
    equipe_nom: str,
    historique_noyau: dict[str, set[int | str]],
) -> float:
    """
    Calcule le pourcentage de joueurs du noyau dans une composition.

    Args:
        composition_ids: Liste des IDs joueurs de la composition
        equipe_nom: Nom de l'equipe
        historique_noyau: {equipe_nom: set(joueur_ids)}

    Returns:
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
    """
    Verifie si la composition respecte la regle du noyau.

    Args:
        composition_ids: Liste des IDs joueurs
        equipe: Equipe concernee
        historique_noyau: {equipe_nom: set(joueur_ids)}
        regles: Regles applicables

    Returns:
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
    else:  # absolu
        return nb_noyau >= noyau_requis


# ==============================================================================
# ZONES D'ENJEU
# ==============================================================================


def calculer_zone_enjeu(
    position: int,
    nb_equipes: int,
    division: str,
) -> str:
    """
    Determine la zone d'enjeu d'une equipe.

    Args:
        position: Position actuelle au classement (1-based)
        nb_equipes: Nombre total d'equipes dans le groupe
        division: Nom de la division

    Returns:
        Zone d'enjeu: "montee" | "course_titre" | "mi_tableau" | "danger" | "descente"
    """
    # Zones de montee
    if position == 1:
        return "montee"

    # Zones de descente
    division_lower = division.lower()
    if "top16" in division_lower or "top 16" in division_lower:
        if position >= 13:
            return "descente"
    elif division_lower in ("n1", "n2"):
        if position >= 9:
            return "descente"
    elif division_lower == "n3":
        if position >= 8:
            return "descente"

    # Zone intermediaire
    if position <= 3:
        return "course_titre"
    elif division_lower == "n3" and position >= 6:
        return "danger"
    elif position >= nb_equipes - 3:
        return "danger"

    return "mi_tableau"


def calculer_ecart_objectif(
    points_equipe: int,
    points_objectif: int,
) -> int:
    """
    Calcule l'ecart par rapport a l'objectif (montee ou maintien).

    Args:
        points_equipe: Points actuels de l'equipe
        points_objectif: Points de l'objectif (1er pour montee, dernier non releguable)

    Returns:
        Ecart en points (positif = en avance, negatif = en retard)
    """
    return points_equipe - points_objectif


# ==============================================================================
# MOUVEMENT DE JOUEURS (VASES COMMUNIQUANTS)
# ==============================================================================


def detecter_mouvement_joueur(
    joueur_elo: int,
    equipe_avant: str,
    equipe_apres: str,
) -> MouvementJoueur:
    """
    Detecte et categorise un mouvement de joueur entre equipes.

    Args:
        joueur_elo: Elo du joueur
        equipe_avant: Nom equipe d'origine
        equipe_apres: Nom equipe destination

    Returns:
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
    elif niveau_apres > niveau_avant:  # Descend (N1 -> N2)
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


# ==============================================================================
# VALIDATION COMPOSITION
# ==============================================================================


def valider_composition(
    composition: list[Joueur],
    equipe: Equipe,
    historique_brulage: dict[int, dict[str, int]],
    historique_noyau: dict[str, set[int]],
    regles: ReglesCompetition,
) -> list[str]:
    """
    Valide une composition selon les regles FFE.

    Args:
        composition: Liste des joueurs alignes
        equipe: Equipe concernee
        historique_brulage: {joueur_id: {equipe_nom: nb_matchs}}
        historique_noyau: {equipe_nom: set(joueur_ids)}
        regles: Regles applicables a cette competition

    Returns:
        Liste des erreurs detectees (vide si valide)
    """
    erreurs: list[str] = []

    if not composition:
        return ["Composition vide"]

    # 3.6.e - Ordre Elo (si obligatoire)
    if regles.get("ordre_elo_obligatoire", True):
        for i in range(len(composition) - 1):
            if composition[i].elo < composition[i + 1].elo - 100:
                erreurs.append(f"Ordre Elo invalide: ech {i + 1}")

    # 3.7.c - Joueur brule
    seuil_brulage = regles.get("seuil_brulage")
    if seuil_brulage is not None:
        for j in composition:
            if est_brule(j.id_fide, equipe.nom, historique_brulage, seuil_brulage):
                erreurs.append(f"{j.nom} est brule pour {equipe.nom}")

    # 3.7.f - Noyau
    composition_ids = [j.id_fide for j in composition]
    if not valide_noyau(composition_ids, equipe, historique_noyau, regles):
        pct = calculer_pct_noyau(composition_ids, equipe.nom, historique_noyau)
        noyau_requis = regles.get("noyau", 0)
        erreurs.append(f"Noyau insuffisant: {pct:.0%} < {noyau_requis}%")

    # 3.7.g - Mutes
    max_mutes = regles.get("max_mutes")
    if max_mutes is not None:
        nb_mutes = sum(1 for j in composition if j.mute)
        if nb_mutes > max_mutes:
            erreurs.append(f"Trop de mutes: {nb_mutes} > {max_mutes}")

    # Elo max (N4 uniquement pour A02, toujours pour C03)
    elo_max = regles.get("elo_max")
    if elo_max is not None:
        # Pour A02, elo_max=2400 ne s'applique qu'en N4
        division_lower = equipe.division.lower()
        is_n4 = "n4" in division_lower or "nationale 4" in division_lower
        # Appliquer si: C03 (Loubatiere) ou N4 A02
        if is_n4 or elo_max < 2400:  # C03 a elo_max=1800
            for j in composition:
                if j.elo > elo_max:
                    erreurs.append(f"{j.nom} Elo {j.elo} > {elo_max}")

    # A02 3.7.h - Quota nationalite (5/8 FR/UE pour A02)
    if regles.get("quota_nationalite", False):
        min_fr_eu = regles.get("min_fr_eu", 0)
        if min_fr_eu > 0:
            # Simplifie: compte FR et UE
            fr_eu = sum(1 for j in composition if j.nationalite in ("FRA", "FR", "UE"))
            if fr_eu < min_fr_eu:
                erreurs.append(f"Quota nationalite: {fr_eu} FR/UE < {min_fr_eu}")

    # Elo total max (Parite)
    elo_total_max = regles.get("elo_total_max")
    if elo_total_max is not None:
        elo_total = sum(j.elo for j in composition)
        if elo_total > elo_total_max:
            erreurs.append(f"Elo total {elo_total} > {elo_total_max}")

    # Composition obligatoire (Parite: 2H + 2F)
    compo_oblig = regles.get("composition_obligatoire")
    if compo_oblig is not None:
        for sexe_key, nb_requis in compo_oblig.items():
            if sexe_key in ("hommes", "garcons"):
                nb = sum(1 for j in composition if j.sexe == Sexe.MASCULIN)
            else:  # femmes, filles
                nb = sum(1 for j in composition if j.sexe == Sexe.FEMININ)
            if nb < nb_requis:
                erreurs.append(f"{sexe_key}: {nb} < {nb_requis}")

    return erreurs


# ==============================================================================
# CALCUL FEATURES POUR DATAFRAME
# ==============================================================================


def calculer_features_joueur(
    joueur_id: int | str,
    equipe_nom: str,
    ronde: int,
    historique_brulage: dict[int | str, dict[str, int]],
    historique_noyau: dict[str, set[int | str]],
    historique_parties: dict[int | str, int],
    regles: ReglesCompetition,
) -> FeaturesReglementaires:
    """
    Calcule les features reglementaires pour un joueur.

    Args:
        joueur_id: ID FIDE du joueur
        equipe_nom: Nom de l'equipe
        ronde: Numero de la ronde
        historique_brulage: {joueur_id: {equipe_nom: nb_matchs}}
        historique_noyau: {equipe_nom: set(joueur_ids)}
        historique_parties: {joueur_id: nb_parties_saison}
        regles: Regles de la competition

    Returns:
        FeaturesReglementaires avec toutes les features calculees
    """
    seuil = regles.get("seuil_brulage", 3)
    max_parties = regles.get("max_parties_saison")

    # Brulage
    brule = (
        est_brule(joueur_id, equipe_nom, historique_brulage, seuil) if seuil is not None else False
    )

    # Matchs dans equipe superieure
    joueur_hist = historique_brulage.get(joueur_id, {})
    niveau_cible = get_niveau_equipe(equipe_nom)
    max_matchs_sup = 0
    for eq_nom, nb_matchs in joueur_hist.items():
        if get_niveau_equipe(eq_nom) < niveau_cible:
            max_matchs_sup = max(max_matchs_sup, nb_matchs)

    matchs_restants = max(0, seuil - max_matchs_sup) if seuil else 0

    # Parties jouees
    nb_parties = historique_parties.get(joueur_id, 0)
    peut_jouer = True
    if max_parties is not None:
        peut_jouer = nb_parties < max_parties
    else:
        # Regle < ronde N
        peut_jouer = nb_parties < ronde

    # Noyau
    est_noyau = joueur_id in get_noyau(equipe_nom, historique_noyau)

    return FeaturesReglementaires(
        joueur_brule=brule,
        matchs_avant_brulage=matchs_restants,
        nb_matchs_joues_saison=nb_parties,
        peut_jouer_ronde_n=peut_jouer,
        est_dans_noyau=est_noyau,
    )


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Types
    "TypeCompetition",
    "NiveauCompetition",
    "Sexe",
    "ReglesCompetition",
    "MouvementJoueur",
    "FeaturesReglementaires",
    "Joueur",
    "Equipe",
    "HistoriqueJoueur",
    # Detection
    "detecter_type_competition",
    "get_niveau_equipe",
    # Regles
    "get_regles_competition",
    "get_regles_a02",
    "get_regles_feminin",
    "get_regles_coupe",
    "get_regles_loubatiere",
    "get_regles_parite",
    "get_regles_jeunes",
    "get_regles_scolaire",
    "get_regles_regionale",
    "get_regles_departemental",
    # Brulage
    "est_brule",
    "matchs_avant_brulage",
    # Noyau
    "get_noyau",
    "calculer_pct_noyau",
    "valide_noyau",
    # Enjeu
    "calculer_zone_enjeu",
    "calculer_ecart_objectif",
    # Mouvement
    "detecter_mouvement_joueur",
    # Validation
    "valider_composition",
    # Features
    "calculer_features_joueur",
]
