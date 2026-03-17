"""Pandera schemas for raw parsed parquets (ISO 5259).

Validates echiquiers.parquet and joueurs.parquet BEFORE feature engineering.
For post-feature validation, see training_schemas.py.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<300 lines, SRP)
"""

import pandas as pd
from pandera import Check, Column, DataFrameSchema


def _check_diff_elo_coherence(df: pd.DataFrame) -> pd.Series:
    """Verify diff_elo == blanc_elo - noir_elo."""
    return df["diff_elo"] == (df["blanc_elo"] - df["noir_elo"])


def _echiquiers_columns() -> dict[str, Column]:
    """Build column definitions for raw echiquiers."""
    return {
        "saison": Column(int, checks=[Check.ge(2002), Check.le(2030)], nullable=False),
        "competition": Column(str, nullable=False),
        "division": Column(str, nullable=False),
        "groupe": Column(str, nullable=False),
        "ronde": Column(int, checks=[Check.ge(1), Check.le(15)], nullable=False),
        "echiquier": Column(int, checks=[Check.ge(1), Check.le(12)], nullable=False),
        "blanc_nom": Column(str, nullable=False),
        "noir_nom": Column(str, nullable=False),
        "blanc_elo": Column(int, checks=[Check.ge(0), Check.le(3000)], nullable=False),
        "noir_elo": Column(int, checks=[Check.ge(0), Check.le(3000)], nullable=False),
        "resultat_blanc": Column(float, checks=Check.isin([0.0, 0.5, 1.0]), nullable=False),
        "resultat_noir": Column(float, checks=Check.isin([0.0, 0.5, 1.0]), nullable=False),
        "type_resultat": Column(str, nullable=False),
        "diff_elo": Column(int, nullable=False),
    }


def _joueurs_columns() -> dict[str, Column]:
    """Build column definitions for raw joueurs."""
    return {
        "nr_ffe": Column(str, checks=Check.str_matches(r"^[A-Z]\d+$"), nullable=False),
        "id_ffe": Column(int, nullable=True),
        "nom": Column(str, nullable=False),
        "prenom": Column(str, nullable=False),
        "nom_complet": Column(str, nullable=True),
        "elo": Column(int, checks=[Check.ge(0), Check.le(3000)], nullable=False),
        "elo_type": Column(str, checks=Check.isin(["F", "N", "E"]), nullable=False),
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
