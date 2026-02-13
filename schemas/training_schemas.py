"""Pandera schemas for ALICE Engine training data validation.

FFE business logic checks and Pandera DataFrameSchema definition.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML (Lineage, Validation)
- ISO/IEC 42001:2023 - AI Management System
"""

import pandas as pd
from pandera import Check, Column, DataFrameSchema

from schemas.training_constants import (
    COMPETITION_TYPES,
    ECHIQUIER_MAX_ABSOLUTE,
    ECHIQUIER_MIN,
    ELO_MAX_N4_PLUS,
    ELO_MAX_REASONABLE,
    ELO_MIN_ESTIME,
    FIDE_TITLES,
    NIVEAU_HIERARCHY_MAX,
    NIVEAU_N4,
    RONDE_MAX_ABSOLUTE,
    RONDE_MIN,
    VALID_GAME_SCORES_ADULTES,
    VALID_GAME_SCORES_ALL,
    VALID_RESULT_SUMS,
    VALID_RESULT_TYPES,
)

# =============================================================================
# FFE BUSINESS LOGIC CHECKS (DataFrame-level)
# =============================================================================


def _check_result_sum_valid(df: pd.DataFrame) -> pd.Series:
    """Check resultat_blanc + resultat_noir is valid.

    Valid sums: 0.0 (ajournement), 1.0 (Adultes/Jeunes 7-8), 2.0 (Jeunes 1-6).
    """
    result_sum = df["resultat_blanc"] + df["resultat_noir"]
    return result_sum.isin(VALID_RESULT_SUMS)


def _check_blanc_equipe_coherence(df: pd.DataFrame) -> pd.Series:
    """Check blanc_equipe is either equipe_dom or equipe_ext."""
    return (df["blanc_equipe"] == df["equipe_dom"]) | (df["blanc_equipe"] == df["equipe_ext"])


def _check_noir_equipe_coherence(df: pd.DataFrame) -> pd.Series:
    """Check noir_equipe is either equipe_dom or equipe_ext."""
    return (df["noir_equipe"] == df["equipe_dom"]) | (df["noir_equipe"] == df["equipe_ext"])


def _check_type_resultat_coherence(df: pd.DataFrame) -> pd.Series:
    """Check type_resultat matches resultat_blanc/noir values."""
    blanc_wins = df["type_resultat"].isin(["victoire_blanc", "victoire_blanc_ajournement"])
    noir_wins = df["type_resultat"].isin(["victoire_noir", "victoire_noir_ajournement"])
    is_nulle = df["type_resultat"] == "nulle"
    is_ajournement = df["type_resultat"] == "ajournement"

    blanc_wins_ok = ~blanc_wins | (df["resultat_blanc"] > df["resultat_noir"])
    noir_wins_ok = ~noir_wins | (df["resultat_noir"] > df["resultat_blanc"])
    nulle_ok = ~is_nulle | (
        (df["resultat_blanc"] == df["resultat_noir"]) & (df["resultat_blanc"] > 0)
    )
    ajournement_ok = ~is_ajournement | ((df["resultat_blanc"] == 0) & (df["resultat_noir"] == 0))

    return blanc_wins_ok & noir_wins_ok & nulle_ok & ajournement_ok


def _check_diff_elo_calculation(df: pd.DataFrame) -> pd.Series:
    """Check diff_elo equals blanc_elo - noir_elo."""
    return df["diff_elo"] == df["blanc_elo"] - df["noir_elo"]


def _check_elo_niveau_restriction(df: pd.DataFrame) -> pd.Series:
    """Check Elo restriction: A02 Art. 3.7.j Elo > 2400 interdit en N4+."""
    is_restricted = (df["niveau"] >= NIVEAU_N4) & (df["niveau"] <= NIVEAU_HIERARCHY_MAX)
    elo_ok = (df["blanc_elo"] <= ELO_MAX_N4_PLUS) & (df["noir_elo"] <= ELO_MAX_N4_PLUS)
    return ~is_restricted | elo_ok


# =============================================================================
# PANDERA SCHEMA DEFINITION
# =============================================================================


def create_training_schema(strict: bool = False) -> DataFrameSchema:
    """Create Pandera schema for training data validation.

    Args:
    ----
        strict: If True, enforce strict FFE regulatory constraints.
                If False, use permissive validation for historical data.

    Returns:
    -------
        DataFrameSchema for validation

    """
    valid_scores = VALID_GAME_SCORES_ADULTES if strict else VALID_GAME_SCORES_ALL

    df_checks = [
        Check(_check_diff_elo_calculation, error="diff_elo must equal blanc_elo - noir_elo"),
    ]

    if strict:
        df_checks.extend(
            [
                Check(
                    _check_result_sum_valid,
                    error="resultat_blanc + resultat_noir must be in [0, 1, 2]",
                ),
                Check(
                    _check_type_resultat_coherence,
                    error="type_resultat must match resultat_blanc/noir values",
                ),
                Check(
                    _check_blanc_equipe_coherence,
                    error="blanc_equipe must be equipe_dom or equipe_ext",
                ),
                Check(
                    _check_noir_equipe_coherence,
                    error="noir_equipe must be equipe_dom or equipe_ext",
                ),
                Check(
                    _check_elo_niveau_restriction,
                    error="A02 Art. 3.7.j: Elo > 2400 interdit en N4+",
                ),
            ]
        )

    return _build_schema(strict, valid_scores, df_checks)


def _build_schema(
    strict: bool, valid_scores: list[float], df_checks: list[Check]
) -> DataFrameSchema:
    """Build the DataFrameSchema with all column definitions."""
    return DataFrameSchema(
        columns={
            **_competition_columns(strict),
            **_match_columns(),
            **_game_columns(valid_scores),
            **_player_columns(strict, "blanc"),
            **_player_columns(strict, "noir"),
            **_reliability_columns(),
            **_form_position_columns(),
            **_multi_team_columns(),
            **_strategic_columns(),
        },
        checks=df_checks,
        coerce=True,
        strict=False,
    )


def _competition_columns(strict: bool) -> dict[str, Column]:
    """Build competition metadata columns."""
    return {
        "saison": Column(int, checks=[Check.ge(2000), Check.le(2100)], nullable=False),
        "competition": Column(str, nullable=False),
        "division": Column(str, nullable=False),
        "groupe": Column(str, nullable=True),
        "ligue": Column(str, nullable=True),
        "ligue_code": Column(str, nullable=True),
        "niveau": Column(int, checks=[Check.ge(0)], nullable=False),
        "type_competition": Column(
            str, checks=Check.isin(COMPETITION_TYPES) if strict else None, nullable=False
        ),
        "ronde": Column(
            int, checks=[Check.ge(RONDE_MIN), Check.le(RONDE_MAX_ABSOLUTE)], nullable=False
        ),
    }


def _match_columns() -> dict[str, Column]:
    """Build match metadata columns."""
    return {
        "equipe_dom": Column(str, nullable=False),
        "equipe_ext": Column(str, nullable=False),
        "score_dom": Column(int, checks=Check.ge(0), nullable=False),
        "score_ext": Column(int, checks=Check.ge(0), nullable=False),
        "date": Column("datetime64[us]", nullable=True),
        "date_str": Column(str, nullable=True),
        "heure": Column(str, nullable=True),
        "jour_semaine": Column(str, nullable=True),
        "lieu": Column(str, nullable=True),
    }


def _game_columns(valid_scores: list[float]) -> dict[str, Column]:
    """Build game data and result columns."""
    return {
        "echiquier": Column(
            int,
            checks=[Check.ge(ECHIQUIER_MIN), Check.le(ECHIQUIER_MAX_ABSOLUTE)],
            nullable=False,
        ),
        "resultat_blanc": Column(float, checks=Check.isin(valid_scores), nullable=False),
        "resultat_noir": Column(float, checks=Check.isin(valid_scores), nullable=False),
        "resultat_text": Column(str, nullable=False),
        "type_resultat": Column(str, checks=Check.isin(VALID_RESULT_TYPES), nullable=False),
        "diff_elo": Column(int, nullable=False),
    }


def _player_columns(strict: bool, color: str) -> dict[str, Column]:
    """Build player columns for a given color (blanc/noir)."""
    return {
        f"{color}_nom": Column(str, nullable=False),
        f"{color}_titre": Column(
            str, checks=Check.isin(FIDE_TITLES) if strict else None, nullable=True
        ),
        f"{color}_elo": Column(
            int, checks=[Check.ge(ELO_MIN_ESTIME), Check.le(ELO_MAX_REASONABLE)], nullable=False
        ),
        f"{color}_equipe": Column(str, nullable=False),
    }


def _reliability_columns() -> dict[str, Column]:
    """Build team and player reliability columns."""
    rc = [Check.ge(0.0), Check.le(1.0)]
    return {
        "taux_forfait_dom": Column(float, checks=rc, nullable=True),
        "taux_non_joue_dom": Column(float, checks=rc, nullable=True),
        "fiabilite_score_dom": Column(float, checks=rc, nullable=True),
        "taux_forfait_ext": Column(float, checks=rc, nullable=True),
        "taux_non_joue_ext": Column(float, checks=rc, nullable=True),
        "fiabilite_score_ext": Column(float, checks=rc, nullable=True),
        "taux_presence_blanc": Column(float, checks=rc, nullable=True),
        "joueur_fantome_blanc": Column(bool, nullable=True),
        "taux_presence_noir": Column(float, checks=rc, nullable=True),
        "joueur_fantome_noir": Column(bool, nullable=True),
    }


def _form_position_columns() -> dict[str, Column]:
    """Build player form and position columns."""
    return {
        "forme_recente_blanc": Column(float, nullable=True),
        "forme_recente_noir": Column(float, nullable=True),
        "echiquier_moyen_blanc": Column(float, nullable=True),
        "echiquier_moyen_noir": Column(float, nullable=True),
    }


def _multi_team_columns() -> dict[str, Column]:
    """Build FFE multi-team feature columns."""
    return {
        "ffe_nb_equipes_blanc": Column(int, checks=Check.ge(0), nullable=True),
        "ffe_niveau_max_blanc": Column(int, nullable=True),
        "ffe_niveau_min_blanc": Column(int, nullable=True),
        "ffe_multi_equipe_blanc": Column(bool, nullable=True),
        "ffe_nb_equipes_noir": Column(int, checks=Check.ge(0), nullable=True),
        "ffe_niveau_max_noir": Column(int, nullable=True),
        "ffe_niveau_min_noir": Column(int, nullable=True),
        "ffe_multi_equipe_noir": Column(bool, nullable=True),
    }


def _strategic_columns() -> dict[str, Column]:
    """Build strategic context columns."""
    return {
        "zone_enjeu_dom": Column(str, nullable=True),
        "niveau_hier_dom": Column(int, nullable=True),
        "zone_enjeu_ext": Column(str, nullable=True),
        "niveau_hier_ext": Column(int, nullable=True),
    }


# Pre-configured schemas
TrainingSchemaPermissive = create_training_schema(strict=False)
TrainingSchemaStrict = create_training_schema(strict=True)
