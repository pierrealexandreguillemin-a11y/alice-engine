"""Types et dataclasses pour Ensemble Stacking - ISO 5055.

Ce module contient les types de donnees pour l'ensemble stacking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from scripts.ml_types import MLClassifier


@dataclass
class StackingMetrics:
    """Metriques du stacking."""

    single_models: dict[str, dict[str, float]]
    stacking_train_auc: float
    stacking_test_auc: float
    soft_voting_test_auc: float
    gain_vs_best_single: float
    gain_vs_soft_voting: float
    best_single_name: str
    best_single_auc: float


@dataclass
class StackingResult:
    """Resultat complet du stacking."""

    meta_model: MLClassifier
    oof_predictions: NDArray[np.float64]
    test_predictions: NDArray[np.float64]
    stacking_test_proba: NDArray[np.float64]
    soft_voting_test_proba: NDArray[np.float64]
    model_weights: dict[str, float]
    model_names: list[str]
    metrics: StackingMetrics
    use_stacking: bool
    n_folds: int
