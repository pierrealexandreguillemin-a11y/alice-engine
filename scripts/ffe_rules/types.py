"""Types et dataclasses FFE - ISO 5055.

Ce module contient les types de donnees pour les regles FFE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict


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
