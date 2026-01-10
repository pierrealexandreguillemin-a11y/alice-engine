"""Module: scripts/comparison/statistical_comparison.py - Model Comparison Pipeline.

Document ID: ALICE-MOD-STATCOMP-001
Version: 1.0.0

Pipeline de comparaison statistique entre modeles ML.
Compare AutoGluon avec l'ensemble baseline CatBoost/XGBoost/LightGBM.

ISO Compliance:
- ISO/IEC 24029:2021 - Statistical validation
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from scripts.comparison.mcnemar_test import McNemarResult, mcnemar_5x2cv_test, mcnemar_simple_test

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    """Resultat de comparaison entre deux modeles.

    Attributes
    ----------
        model_a_name: Nom du modele A
        model_b_name: Nom du modele B
        mcnemar_result: Resultat du test McNemar
        metrics_a: Metriques du modele A
        metrics_b: Metriques du modele B
        winner: Modele gagnant
        practical_significance: True si difference pratiquement significative
        recommendation: Recommandation basee sur l'analyse
    """

    model_a_name: str
    model_b_name: str
    mcnemar_result: McNemarResult
    metrics_a: dict[str, float]
    metrics_b: dict[str, float]
    winner: str
    practical_significance: bool
    recommendation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


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
    if isinstance(y_test, pd.Series):
        y_array = y_test.values
    else:
        y_array = y_test

    pred_a = np.array(model_a_predict(X_test))
    pred_b = np.array(model_b_predict(X_test))

    # Test McNemar
    mcnemar = mcnemar_simple_test(y_array, pred_a, pred_b, alpha=alpha)

    # Calculer les metriques
    metrics_a = _compute_metrics(y_array, pred_a, model_a_predict_proba, X_test)
    metrics_b = _compute_metrics(y_array, pred_b, model_b_predict_proba, X_test)

    # Determiner le gagnant
    if mcnemar.significant:
        winner = model_a_name if mcnemar.winner == "model_a" else model_b_name
    else:
        # Pas de difference significative -> tie
        winner = "tie"

    # Significativite pratique
    acc_diff = abs(metrics_a["accuracy"] - metrics_b["accuracy"])
    practical_significance = acc_diff >= effect_threshold

    # Generer la recommandation
    recommendation = _generate_recommendation(
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


def save_comparison_report(
    comparison: ModelComparison,
    output_path: Path,
) -> None:
    """Sauvegarde le rapport de comparaison.

    Args:
    ----
        comparison: Resultat de la comparaison
        output_path: Chemin du fichier JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convertir en dict serializable
    report = {
        "model_a_name": comparison.model_a_name,
        "model_b_name": comparison.model_b_name,
        "winner": comparison.winner,
        "practical_significance": comparison.practical_significance,
        "recommendation": comparison.recommendation,
        "timestamp": comparison.timestamp,
        "mcnemar": {
            "statistic": comparison.mcnemar_result.statistic,
            "p_value": comparison.mcnemar_result.p_value,
            "significant": comparison.mcnemar_result.significant,
            "effect_size": comparison.mcnemar_result.effect_size,
            "confidence_interval": comparison.mcnemar_result.confidence_interval,
        },
        "metrics_a": comparison.metrics_a,
        "metrics_b": comparison.metrics_b,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Comparison report saved to {output_path}")


def _compute_metrics(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    predict_proba: Callable | None,
    X: Any,
) -> dict[str, float]:
    """Calcule les metriques de performance.

    Args:
    ----
        y_true: Labels vrais
        y_pred: Predictions
        predict_proba: Fonction predict_proba (optionnel)
        X: Features

    Returns:
    -------
        Dict des metriques
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
    }

    # AUC-ROC si probas disponibles
    if predict_proba is not None:
        try:
            proba = predict_proba(X)
            if len(proba.shape) > 1:
                proba = proba[:, 1]
            metrics["auc_roc"] = float(roc_auc_score(y_true, proba))
        except Exception as e:
            logger.warning(f"Could not compute AUC-ROC: {e}")

    return metrics


def _generate_recommendation(
    winner: str,
    mcnemar: McNemarResult,
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    model_a_name: str,
    model_b_name: str,
    practical_significance: bool,
) -> str:
    """Genere une recommandation basee sur l'analyse.

    ISO 42001: Decision documentee et justifiee.
    """
    if winner == "tie":
        if practical_significance:
            better = model_a_name if metrics_a["accuracy"] > metrics_b["accuracy"] else model_b_name
            return (
                f"Pas de difference statistiquement significative (p={mcnemar.p_value:.3f}), "
                f"mais {better} montre une tendance. Considerer d'autres facteurs "
                f"(inference time, interpretabilite)."
            )
        return (
            f"Pas de difference significative entre les modeles (p={mcnemar.p_value:.3f}). "
            f"Choisir selon criteres operationnels (vitesse, maintenance)."
        )

    if practical_significance:
        return (
            f"{winner} est significativement meilleur (p={mcnemar.p_value:.3f}) "
            f"avec une difference pratiquement significative. "
            f"Recommandation: deployer {winner}."
        )

    return (
        f"{winner} est statistiquement meilleur (p={mcnemar.p_value:.3f}) "
        f"mais la difference pratique est faible. "
        f"Considerer les couts operationnels avant decision."
    )


def full_comparison_pipeline(
    model_a_fit: Callable,
    model_b_fit: Callable,
    model_a_predict: Callable,
    model_b_predict: Callable,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    model_a_name: str = "AutoGluon",
    model_b_name: str = "Baseline",
    n_iterations: int = 5,
    output_dir: Path | None = None,
) -> ModelComparison:
    """Pipeline complet avec McNemar 5x2cv.

    Args:
    ----
        model_a_fit: Fonction fit modele A
        model_b_fit: Fonction fit modele B
        model_a_predict: Fonction predict modele A
        model_b_predict: Fonction predict modele B
        X: Features
        y: Labels
        model_a_name: Nom modele A
        model_b_name: Nom modele B
        n_iterations: Iterations McNemar 5x2cv
        output_dir: Repertoire de sortie

    Returns:
    -------
        ModelComparison avec resultats complets

    ISO 24029/42001: Pipeline de comparaison complet et tracable.
    """
    logger.info(f"Starting full comparison: {model_a_name} vs {model_b_name}")

    # McNemar 5x2cv
    mcnemar = mcnemar_5x2cv_test(
        model_a_fit=model_a_fit,
        model_b_fit=model_b_fit,
        model_a_predict=model_a_predict,
        model_b_predict=model_b_predict,
        X=X,
        y=y,
        n_iterations=n_iterations,
    )

    # Determiner gagnant
    if mcnemar.significant:
        winner = model_a_name if mcnemar.winner == "model_a" else model_b_name
    else:
        winner = "tie"

    # Significativite pratique
    acc_diff = abs(mcnemar.model_a_mean_accuracy - mcnemar.model_b_mean_accuracy)
    practical_significance = acc_diff >= 0.05

    comparison = ModelComparison(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        mcnemar_result=mcnemar,
        metrics_a={"accuracy": mcnemar.model_a_mean_accuracy},
        metrics_b={"accuracy": mcnemar.model_b_mean_accuracy},
        winner=winner,
        practical_significance=practical_significance,
        recommendation=_generate_recommendation(
            winner=winner,
            mcnemar=mcnemar,
            metrics_a={"accuracy": mcnemar.model_a_mean_accuracy},
            metrics_b={"accuracy": mcnemar.model_b_mean_accuracy},
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            practical_significance=practical_significance,
        ),
    )

    # Sauvegarder si repertoire specifie
    if output_dir:
        save_comparison_report(comparison, output_dir / "comparison_report.json")

    return comparison
