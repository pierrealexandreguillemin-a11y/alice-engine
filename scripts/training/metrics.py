"""Calcul des métriques pour l'entraînement - ISO 5055/25010.

Ce module contient les fonctions de calcul des métriques.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scripts.ml_types import ModelMetrics

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def compute_all_metrics(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    y_proba: NDArray[np.float64],
) -> ModelMetrics:
    """Calcule toutes les metriques conformes ISO 25010."""
    cm = confusion_matrix(y_true, y_pred)

    return ModelMetrics(
        auc_roc=float(roc_auc_score(y_true, y_proba)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
        log_loss=float(log_loss(y_true, y_proba)),
        true_negatives=int(cm[0, 0]),
        false_positives=int(cm[0, 1]),
        false_negatives=int(cm[1, 0]),
        true_positives=int(cm[1, 1]),
    )
