"""Dataclasses pour parsing dataset FFE - ISO 5055.

Ce module contient les structures de données:
- Joueur (partie individuelle)
- Echiquier
- Match
- Metadata
- JoueurLicencie (liste FFE)

Conformité ISO/IEC 5055, 25010.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Joueur:
    """Representation d'un joueur."""

    nom: str
    prenom: str
    nom_complet: str
    elo: int
    titre: str = ""
    titre_fide: str = ""


@dataclass
class Echiquier:
    """Representation d'une partie sur un echiquier."""

    numero: int
    blanc: Joueur
    noir: Joueur
    equipe_blanc: str
    equipe_noir: str
    resultat_blanc: float
    resultat_noir: float
    resultat_text: str
    type_resultat: str
    diff_elo: int


@dataclass
class Match:
    """Representation d'un match entre deux equipes."""

    ronde: int
    equipe_dom: str
    equipe_ext: str
    score_dom: int
    score_ext: int
    date: datetime | None = None
    date_str: str = ""
    heure: str = ""
    jour_semaine: str = ""
    lieu: str = ""
    echiquiers: list[Echiquier] = field(default_factory=list)


@dataclass
class Metadata:
    """Metadonnees extraites du chemin."""

    saison: int
    competition: str
    division: str
    groupe: str
    ligue: str = ""
    ligue_code: str = ""
    niveau: int = 0
    type_competition: str = ""


@dataclass
class JoueurLicencie:
    """Joueur licencie FFE (depuis pages players)."""

    nr_ffe: str  # Licence FFE (ex: K59857)
    id_ffe: int  # ID interne FFE
    nom: str
    prenom: str
    nom_complet: str
    affiliation: str  # A, B ou N (type de licence FFE)
    elo: int
    elo_type: str  # F=Fide, N=National, E=Estime
    elo_rapide: int
    elo_rapide_type: str
    elo_blitz: int
    elo_blitz_type: str
    categorie: str  # SenM, MinM, etc. (format legacy)
    mute: bool  # Mute = transfere d'un autre club cette saison
    club: str  # Nom du club (parfois + ville)
