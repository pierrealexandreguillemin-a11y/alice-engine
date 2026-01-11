"""Comparaison de modeles - ISO 24029.

Ce module contient la fonction principale de comparaison.

ISO Compliance:
- ISO/IEC 24029:2021 - Statistical validation
- ISO/IEC 5055:2021 - Code Quality (<100 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scripts.comparison.mcnemar_test import McNemarResult, mcnemar_simple_test
from scripts.comparison.metrics import compute_metrics
from scripts.comparison.recommendation import generate_recommendation
from scripts.comparison.types import ModelComparison

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compare_models(
    model_a_predict: Callable,
    model_b_predict: Callable,
    X_test: NDArray[np.float64] | pd.DataFrame,
    y_test: NDArray[np.int64] | pd.Series,
    model_a_predict_proba: Callable | None = None,
    model_b_predict_proba: Callable | None = None,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    alpha: float = 0.05,
    effect_threshold: float = 0.05,
) -> ModelComparison:
    """Compare deux modeles avec test McNemar simple.

    Args:
    ----
        model_a_predict: Fonction predict pour modele A
        model_b_predict: Fonction predict pour modele B
        X_test: Features de test
        y_test: Labels de test
        model_a_predict_proba: Fonction predict_proba pour A (optionnel)
        model_b_predict_proba: Fonction predict_proba pour B (optionnel)
        model_a_name: Nom du modele A
        model_b_name: Nom du modele B
        alpha: Seuil de significativite
        effect_threshold: Seuil pour significativite pratique

    Returns:
    -------
        ModelComparison avec resultats detailles

    ISO 24029: Comparaison statistique robuste.
    """
    logger.info(f"Comparing {model_a_name} vs {model_b_name}")

    # Predictions
    y_array = y_test.values if isinstance(y_test, pd.Series) else y_test

    pred_a = np.array(model_a_predict(X_test))
    pred_b = np.array(model_b_predict(X_test))

    # Test McNemar
    mcnemar = mcnemar_simple_test(y_array, pred_a, pred_b, alpha=alpha)

    # Calculer les metriques
    metrics_a = compute_metrics(y_array, pred_a, model_a_predict_proba, X_test)
    metrics_b = compute_metrics(y_array, pred_b, model_b_predict_proba, X_test)

    # Determiner le gagnant
    winner = _determine_winner(mcnemar, model_a_name, model_b_name)

    # Significativite pratique
    acc_diff = abs(metrics_a["accuracy"] - metrics_b["accuracy"])
    practical_significance = acc_diff >= effect_threshold

    # Generer la recommandation
    recommendation = generate_recommendation(
        winner=winner,
        mcnemar=mcnemar,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        practical_significance=practical_significance,
    )

    return ModelComparison(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        mcnemar_result=mcnemar,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        winner=winner,
        practical_significance=practical_significance,
        recommendation=recommendation,
    )


def _determine_winner(mcnemar: McNemarResult, model_a_name: str, model_b_name: str) -> str:
    """Determine le modele gagnant."""
    if mcnemar.significant:
        return model_a_name if mcnemar.winner == "model_a" else model_b_name
    return "tie"
