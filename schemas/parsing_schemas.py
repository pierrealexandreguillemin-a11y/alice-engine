"""Pandera schemas for raw parsed parquets (ISO 5259).

Validates echiquiers.parquet and joueurs.parquet BEFORE feature engineering.
For post-feature validation, see training_schemas.py.

Uses FFE regulatory constants from training_constants.py — same source of
truth as the training schemas.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<300 lines, SRP)
"""

import pandas as pd
from pandera import Check, Column, DataFrameSchema

from schemas.training_constants import (
    ECHIQUIER_MAX_ABSOLUTE,
    ECHIQUIER_MIN,
    ELO_MAX_REASONABLE,
    RONDE_MAX_ABSOLUTE,
    RONDE_MIN,
    VALID_GAME_SCORES_ALL,
    VALID_RESULT_TYPES,
)

# Raw data includes non-played and unparseable results
_RAW_RESULT_TYPES = [
    *VALID_RESULT_TYPES,
    "non_joue",
    "forfait_blanc",
    "forfait_noir",
    "double_forfait",
    "inconnu",
]


def _check_diff_elo_coherence(df: pd.DataFrame) -> pd.Series:
    """Verify diff_elo == blanc_elo - noir_elo."""
    return df["diff_elo"] == (df["blanc_elo"] - df["noir_elo"])


def _echiquiers_columns() -> dict[str, Column]:
    """Build column definitions for raw echiquiers.

    Elo bounds: [0, ELO_MAX_REASONABLE]. 0 = no Elo (unrated player).
    Played games have Elo >= ELO_MIN_ESTIME (799), but raw data includes
    forfaits/non-joues where Elo may be 0.
    """
    return {
        "saison": Column(int, checks=[Check.ge(2002), Check.le(2030)], nullable=False),
        "competition": Column(str, nullable=False),
        "division": Column(str, nullable=False),
        "groupe": Column(str, nullable=False),
        "ronde": Column(
            int,
            checks=[Check.ge(RONDE_MIN), Check.le(RONDE_MAX_ABSOLUTE)],
            nullable=False,
        ),
        "echiquier": Column(
            int,
            checks=[Check.ge(ECHIQUIER_MIN), Check.le(ECHIQUIER_MAX_ABSOLUTE)],
            nullable=False,
        ),
        "blanc_nom": Column(str, nullable=False),
        "noir_nom": Column(str, nullable=False),
        "blanc_elo": Column(
            int,
            checks=[Check.ge(0), Check.le(ELO_MAX_REASONABLE)],
            nullable=False,
        ),
        "noir_elo": Column(
            int,
            checks=[Check.ge(0), Check.le(ELO_MAX_REASONABLE)],
            nullable=False,
        ),
        "resultat_blanc": Column(
            float,
            checks=Check.isin(VALID_GAME_SCORES_ALL),
            nullable=False,
        ),
        "resultat_noir": Column(
            float,
            checks=Check.isin(VALID_GAME_SCORES_ALL),
            nullable=False,
        ),
        "type_resultat": Column(str, checks=Check.isin(_RAW_RESULT_TYPES), nullable=False),
        "diff_elo": Column(int, nullable=False),
    }


def _joueurs_columns() -> dict[str, Column]:
    """Build column definitions for raw joueurs.

    Elo: 0 = no Elo (3 players). Real Elo >= ELO_MIN_ESTIME (799).
    elo_type: F=FIDE, N=National, E=Estimated, ""=unknown (3 players).
    """
    return {
        "nr_ffe": Column(str, checks=Check.str_matches(r"^[A-Z]\d+$"), nullable=False),
        "id_ffe": Column(int, nullable=True),
        "nom": Column(str, nullable=False),
        "prenom": Column(str, nullable=False),
        "nom_complet": Column(str, nullable=True),
        "elo": Column(
            int,
            checks=[Check.ge(0), Check.le(ELO_MAX_REASONABLE)],
            nullable=False,
        ),
        "elo_type": Column(str, checks=Check.isin(["F", "N", "E", ""]), nullable=False),
        "categorie": Column(str, nullable=False),
        "club": Column(str, nullable=False),
    }


EchiquiersRawSchema = DataFrameSchema(
    columns=_echiquiers_columns(),
    checks=[Check(_check_diff_elo_coherence, error="diff_elo != blanc_elo - noir_elo")],
    coerce=False,
)

JoueursRawSchema = DataFrameSchema(
    columns=_joueurs_columns(),
    coerce=False,
)
