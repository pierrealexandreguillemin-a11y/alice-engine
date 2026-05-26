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

    Extension Plan 2 Task 9 (D-P3-11) : champs legacy-migration pour
    permettre a RuleEngine de couvrir articles 3.7.c (brule),
    3.7.d (same_group), 3.7.e (match_count), 3.7.h (foreign_quota),
    3.7.i (fr_gender). Defaults backward-compat preserves.
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

    # Plan 2 legacy extension (D-P3-11 migration RuleEngine)
    matchs_joues: int = 0
    matchs_equipe_sup: tuple[tuple[str, int, int], ...] | None = None
    group_history: str | None = None
    is_french_eu: bool = True
    is_french: bool = True
    sexe: str = "M"


@dataclass(frozen=True)
class CompetitionContext:
    """Contexte d'un match pour appliquer les bonnes regles FFE.

    Porte les parametres competition-specifiques (noyau, max_mutes,
    elo_max) qui surchargent ou completent les `conditions` des Rules.

    Extension Plan 2 Task 9 (D-P3-11) : champs pour permettre a
    RuleEngine d'evaluer les articles 3.7.c (brule), 3.7.d
    (same_group), 3.7.f (noyau), 3.7.h (foreign_quota). Defaults
    backward-compat preserves (target_team_id non restrictif,
    noyau vide -> check_noyau retourne True).
    """

    competition_code: str
    niveau: str
    ronde: int
    team_size: int
    noyau_min: int
    max_mutes: int
    elo_max: int | None
    target_team_id: str = "_1"
    target_team_rank: int = 1
    target_group: str = "default"
    noyau: frozenset[str] = frozenset()
    brule_seuil: int = 3
    min_fr_eu: int = 5


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


@dataclass(frozen=True)
class TeamSpec:
    """A team specification for multi-team CE-adverse simulation (Phase 4a).

    Used by AdverseCESolver to model the N teams of an opponent club
    fielding on the same weekend. `target_team=True` marks the team
    whose composition the ML pipeline is predicting downstream.

    Document ID: ALICE-ALI-TEAMSPEC
    Version: 1.0.0
    Count: N teams per club
    """

    team_name: str
    division: str
    board_count: int
    target_team: bool = False


@dataclass(frozen=True)
class AdverseCESolution:
    """Result of CE-adverse OR-Tools solve for one team (Phase 4a).

    Returned by AdverseCESolver._solve_one_team per A02 §3.7.b top-down
    iteration. `assignments` is a tuple of `(player_nr_ffe, board_idx)`
    pairs (frozen for hashability + ISO 5055 SRP).

    Document ID: ALICE-ALI-ADVERSE-CE-SOLUTION
    Version: 1.0.0
    Count: 1 per team
    """

    team_name: str
    assignments: tuple[tuple[str, int], ...]
    solver_status: str
    wall_time_ms: int
    objective_value: float


@dataclass(frozen=True)
class PreferenceFeatures:
    """Features per (player, team) pair for PreferenceModel (Phase 4a T2).

    Consumed by `services.ali.preference_model.PreferenceModel.predict_proba`
    to score `P(player -> team_rank | Elo, recency F2, streak F3, brule, hist)`.
    Frozen for hashability and ISO 29119 deterministic test invariants.

    Document ID: ALICE-ALI-PREF-FEATURES
    Version: 1.0.0
    Count: 1 per (player, team) observation
    """

    player_nr_ffe: str
    team_name: str
    elo: int
    recency_decay: float  # F2 recency (decay_lambda=0.9 over rounds)
    streak_count: int  # F3 consecutive participations (MVP placeholder)
    brule_count: int  # A02 §3.7.c history count higher teams (MVP placeholder)
    historical_team_rank: int  # MVP placeholder = 0 (Phase 4a); using observed rank = label leakage
