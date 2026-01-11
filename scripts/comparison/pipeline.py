"""Pipeline de comparaison complet - ISO 24029/42001.

Ce module contient le pipeline McNemar 5x2cv complet.

ISO Compliance:
- ISO/IEC 24029:2021 - Statistical validation
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 5055:2021 - Code Quality (<100 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.comparison.mcnemar_test import McNemarResult, mcnemar_5x2cv_test
from scripts.comparison.recommendation import generate_recommendation
from scripts.comparison.report import save_comparison_report
from scripts.comparison.types import ModelComparison

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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
    winner = _determine_winner(mcnemar, model_a_name, model_b_name)

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
        recommendation=generate_recommendation(
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


def _determine_winner(mcnemar: McNemarResult, model_a_name: str, model_b_name: str) -> str:
    """Determine le modele gagnant."""
    if mcnemar.significant:
        return model_a_name if mcnemar.winner == "model_a" else model_b_name
    return "tie"
