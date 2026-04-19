"""Frozen dataclasses pour ALI (ISO 29119 testability, ISO 5055 SRP).

Ces types representent les entrees/sorties pures des dispatchers RuleEngine
(`filter_candidates`, `validate_lineup`). Aucun comportement, pas de logique
metier ici : simples porteurs de donnees immuables pour faciliter tests
et tracabilite ISO 42001.

Document ID: ALICE-ALI-TYPES
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PlayerCandidate:
    """Un joueur eligible du pool (enrichi ou brut).

    Champs obligatoires : identite FFE + attributs d'eligibilite
    (mute, licence_active, categorie, genre) consommes par les
    regles FFE 3.6.e, 3.7.a, 3.7.g, 3.7.j.

    Champs optionnels : features historiques (brule) non requises
    pour les regles de base. Alimentes par le feature_store en Phase 3.
    """

    nr_ffe: str
    nom: str
    prenom: str
    elo: int
    club: str
    mute: bool
    genre: str
    categorie: str
    licence_active: bool
    age_min: int | None = None
    age_max: int | None = None

    taux_presence_effectif: float | None = None
    played_lag1: bool | None = None
    played_lag2: bool | None = None
    played_lag3: bool | None = None
    echiquier_prefere: int | None = None
    role_type: str | None = None


@dataclass(frozen=True)
class CompetitionContext:
    """Contexte d'un match pour appliquer les bonnes regles FFE.

    Porte les parametres competition-specifiques (noyau, max_mutes,
    elo_max) qui surchargent ou completent les `conditions` des Rules.
    """

    competition_code: str
    niveau: str
    ronde: int
    team_size: int
    noyau_min: int
    max_mutes: int
    elo_max: int | None


@dataclass(frozen=True)
class RuleViolation:
    """Une violation de regle detectee lors de validate_lineup.

    Porte le `rule_uuid` (ISO 42001 canonical traceability) et
    `rule_article` (ISO 42005 impact : ref FFE humaine) pour audit.
    """

    rule_uuid: str
    rule_article: str
    message: str
    severity: Literal["error", "warning"]
