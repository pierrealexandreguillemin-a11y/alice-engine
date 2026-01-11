"""Calcul des metriques de comparaison - ISO 24029.

Ce module contient les fonctions de calcul de metriques.

ISO Compliance:
- ISO/IEC 24029:2021 - Statistical validation
- ISO/IEC 5055:2021 - Code Quality (<50 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_metrics(
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
