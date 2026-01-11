"""Comparaison avec baseline - ISO 42001.

Ce module contient la fonction de comparaison AutoGluon vs baseline.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 5055:2021 - Code Quality (<80 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from scripts.comparison.core import compare_models
from scripts.comparison.report import save_comparison_report
from scripts.comparison.types import ModelComparison

if TYPE_CHECKING:
    import numpy as np
    from autogluon.tabular import TabularPredictor
    from numpy.typing import NDArray


def compare_with_baseline(
    autogluon_predictor: TabularPredictor,
    baseline_predict: Callable,
    test_data: pd.DataFrame,
    label: str,
    baseline_predict_proba: Callable | None = None,
    output_path: Path | None = None,
) -> ModelComparison:
    """Compare AutoGluon avec le modele baseline.

    Args:
    ----
        autogluon_predictor: TabularPredictor AutoGluon
        baseline_predict: Fonction predict du baseline
        test_data: Donnees de test avec label
        label: Nom de la colonne cible
        baseline_predict_proba: Fonction predict_proba du baseline
        output_path: Chemin pour sauvegarder le rapport

    Returns:
    -------
        ModelComparison avec resultats detailles

    ISO 42001: Comparaison tracable.
    """
    X_test = test_data.drop(columns=[label])
    y_test = test_data[label]

    # Wrapper pour AutoGluon predict
    def ag_predict(X: pd.DataFrame) -> NDArray[np.int64]:
        return autogluon_predictor.predict(X).values

    def ag_predict_proba(X: pd.DataFrame) -> NDArray[np.float64]:
        proba = autogluon_predictor.predict_proba(X)
        if isinstance(proba, pd.DataFrame):
            return proba.values
        return proba

    comparison = compare_models(
        model_a_predict=ag_predict,
        model_b_predict=baseline_predict,
        X_test=X_test,
        y_test=y_test,
        model_a_predict_proba=ag_predict_proba,
        model_b_predict_proba=baseline_predict_proba,
        model_a_name="AutoGluon",
        model_b_name="Baseline (CatBoost/XGB/LGB)",
    )

    # Sauvegarder si chemin specifie
    if output_path:
        save_comparison_report(comparison, output_path)

    return comparison
