"""Detection et regles par competition FFE - ISO 5055.

Ce module contient la detection du type de competition
et les regles applicables par type.
"""

from __future__ import annotations

from scripts.ffe_rules.types import ReglesCompetition, TypeCompetition

# Table de detection par patterns (ordre = priorite)
_COMPETITION_PATTERNS: list[tuple[list[str], TypeCompetition]] = [
    # Coupes speciales (prioritaire)
    (["loubatiere", "loubatière"], TypeCompetition.C03),
    (["parite", "parité"], TypeCompetition.C04),
    # Feminin
    (["feminin", "féminin", "12f", "feminine"], TypeCompetition.F01),
    # Scolaire
    (["scolaire", "etablissement", "école"], TypeCompetition.J03),
    # Jeunes
    (["jeune", "junior", "top jeunes"], TypeCompetition.J02),
    # Regional
    (["regionale", "régionale", "r1 ", "r2 ", "r3 "], TypeCompetition.REG),
    # Departemental
    (["departemental", "départemental", "n5", "n6", "criterium"], TypeCompetition.DEP),
]


def _match_coupe_france(nom: str) -> bool:
    """Detecte la Coupe de France (pattern combine)."""
    return "coupe" in nom and "france" in nom


def detecter_type_competition(nom_competition: str) -> TypeCompetition:
    """Detecte le type de competition pour appliquer les bonnes regles.

    Args:
    ----
        nom_competition: Nom de la competition (ex: "Nationale 2", "Coupe Loubatiere")

    Returns:
    -------
        TypeCompetition enum correspondant

    ISO 5055: Dispatch table pattern (complexite B).
    """
    nom = nom_competition.lower()

    # Patterns simples via table
    for patterns, comp_type in _COMPETITION_PATTERNS:
        if any(p in nom for p in patterns):
            return comp_type

    # Pattern combine Coupe de France
    if _match_coupe_france(nom):
        return TypeCompetition.C01

    # National (default)
    return TypeCompetition.A02


def get_niveau_equipe(equipe: str) -> int:
    """Retourne le niveau hierarchique d'une equipe (1=Top16, 8=N4).

    Args:
    ----
        equipe: Nom de l'equipe incluant sa division

    Returns:
    -------
        int: Niveau (1=plus fort, 10=plus faible)
    """
    equipe_lower = equipe.lower()
    niveaux: dict[str, int] = {
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
        "top12f": 1,
        "top 12f": 1,
        "n1f": 2,
        "n2f": 3,
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
    return 10


def get_regles_a02() -> ReglesCompetition:
    """Regles du Championnat de France des Clubs (A02)."""
    return ReglesCompetition(
        taille_equipe=8,
        seuil_brulage=3,
        max_parties_saison=None,
        max_mutes=3,
        min_fr_eu=5,
        ordre_elo_obligatoire=True,
        elo_max=2400,
        noyau=50,
        noyau_type="pourcentage",
        quota_nationalite=True,
    )


def get_regles_feminin() -> ReglesCompetition:
    """Regles du Championnat Feminin (F01)."""
    return ReglesCompetition(
        taille_equipe=4,
        seuil_brulage=1,
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
        ordre_elo_obligatoire=False,
        quota_nationalite=True,
    )


def get_regles_loubatiere() -> ReglesCompetition:
    """Regles de la Coupe Loubatiere (C03) - bas Elo."""
    return ReglesCompetition(
        taille_equipe=4,
        elo_max=1800,
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
        ordre_elo_obligatoire=False,
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
        ordre_elo_obligatoire=False,
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
    """Retourne les regles applicables selon le type de competition.

    Args:
    ----
        type_comp: Type de competition (enum TypeCompetition)

    Returns:
    -------
        ReglesCompetition avec toutes les regles applicables
    """
    regles_map: dict[TypeCompetition, ReglesCompetition] = {
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
    return regles_map.get(type_comp, get_regles_a02())
