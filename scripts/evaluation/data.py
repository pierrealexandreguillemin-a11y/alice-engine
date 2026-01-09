"""Data preparation for model evaluation.

This module handles feature preparation and label encoding
for the evaluation pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from scripts.evaluation.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def prepare_features(
    df: pd.DataFrame,
    label_encoders: dict | None = None,
    fit_encoders: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict]:
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
    # Convertir en classification binaire: victoire (1) vs non-victoire (0)
    y = (df["resultat_blanc"] == 1.0).astype(int)

    # Features numeriques
    X_numeric = df[NUMERIC_FEATURES].fillna(0)

    # Features categorielles - encodage
    if label_encoders is None:
        label_encoders = {}

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
            le = label_encoders.get(col)
            if le is None:
                continue
            # Gerer les valeurs inconnues (capture via default arg to avoid B023)
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x, kc=known_classes: x if x in kc else "UNKNOWN")
            if "UNKNOWN" not in known_classes:
                le.classes_ = np.append(le.classes_, "UNKNOWN")
            X_cat_encoded[col] = le.transform(df[col])

    # Combiner
    X = pd.concat([X_numeric.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)], axis=1)

    return X, y, label_encoders
