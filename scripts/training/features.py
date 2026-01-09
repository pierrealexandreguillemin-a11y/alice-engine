"""Préparation des features pour l'entraînement - ISO 5055/5259.

Ce module contient les fonctions de préparation des features.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML (missing data flags)
- ISO/IEC 5055 - Code Quality

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.preprocessing import LabelEncoder

NUMERIC_FEATURES: list[str] = [
    "blanc_elo",
    "noir_elo",
    "diff_elo",
    "echiquier",
    "niveau",
    "ronde",
]

CATEGORICAL_FEATURES: list[str] = [
    "type_competition",
    "division",
    "ligue_code",
    "blanc_titre",
    "noir_titre",
    "jour_semaine",
]


def prepare_features(
    df: pd.DataFrame,
    label_encoders: dict[str, LabelEncoder] | None = None,
    *,
    fit_encoders: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    """Prepare les features pour l'entrainement.

    Args:
    ----
        df: DataFrame brut
        label_encoders: encodeurs existants (pour valid/test)
        fit_encoders: True pour train, False pour valid/test

    Returns:
    -------
        X, y, label_encoders
    """
    df = df.copy()

    # Target: resultat blanc (0=defaite, 0.5=nulle, 1=victoire)
    y = (df["resultat_blanc"] == 1.0).astype(int)

    # Features numeriques avec flags qualité (ISO 5259)
    X_numeric, missing_flags = _prepare_numeric_with_quality_flags(df)

    # Features categorielles - encodage
    if label_encoders is None:
        label_encoders = {}

    X_cat_encoded = _encode_categorical_features(df, label_encoders, fit_encoders=fit_encoders)

    # Combiner (ISO 5259: inclure flags qualité données)
    X = pd.concat(
        [
            X_numeric.reset_index(drop=True),
            missing_flags.reset_index(drop=True),
            X_cat_encoded.reset_index(drop=True),
        ],
        axis=1,
    )

    return X, y, label_encoders


def _prepare_numeric_with_quality_flags(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prépare features numériques avec flags qualité (ISO 5259).

    Au lieu d'imputation silencieuse, on crée des flags explicites
    pour tracer les données manquantes (Data Quality for ML).

    Returns
    -------
        X_numeric: features avec valeurs imputées
        missing_flags: flags binaires indiquant données manquantes
    """
    X_numeric = df[NUMERIC_FEATURES].copy()
    missing_flags = pd.DataFrame()

    # Créer flag pour chaque feature avec données manquantes
    for col in NUMERIC_FEATURES:
        if X_numeric[col].isna().any():
            missing_flags[f"{col}_missing"] = X_numeric[col].isna().astype(int)

    # Imputer avec médiane (plus robuste que 0) - ISO 5259 recommandation
    for col in NUMERIC_FEATURES:
        if X_numeric[col].isna().any():
            median_val = X_numeric[col].median()
            X_numeric[col] = X_numeric[col].fillna(median_val)

    return X_numeric, missing_flags


def _encode_categorical_features(
    df: pd.DataFrame,
    label_encoders: dict[str, LabelEncoder],
    *,
    fit_encoders: bool,
) -> pd.DataFrame:
    """Encode les features catégorielles."""
    from sklearn.preprocessing import LabelEncoder  # Lazy import

    X_cat_encoded = pd.DataFrame()

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue

        df[col] = df[col].fillna("UNKNOWN").astype(str)

        if fit_encoders:
            le = LabelEncoder()
            X_cat_encoded[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            encoded = _encode_with_existing_encoder(df, col, label_encoders)
            if encoded is not None:
                X_cat_encoded[col] = encoded

    return X_cat_encoded


def _encode_with_existing_encoder(
    df: pd.DataFrame,
    col: str,
    label_encoders: dict[str, LabelEncoder],
) -> pd.Series | None:
    """Encode une colonne avec un encodeur existant."""
    le = label_encoders.get(col)
    if le is None:
        return None

    known_classes = set(le.classes_)
    values = df[col].apply(lambda x, kc=known_classes: x if x in kc else "UNKNOWN")

    if "UNKNOWN" not in known_classes:
        le.classes_ = np.append(le.classes_, "UNKNOWN")

    return pd.Series(le.transform(values))
