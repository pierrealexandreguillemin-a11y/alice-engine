"""Evaluation metrics for model assessment.

This module contains functions for computing model performance metrics.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from sklearn.metrics import accuracy_score, roc_auc_score

if TYPE_CHECKING:
    import pandas as pd


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
) -> dict:
    """Evalue un modele et retourne les metriques.

    Args:
    ----
        model: Trained model with predict_proba method
        X: Feature matrix
        y: True labels
        name: Model name for identification

    Returns:
    -------
        Dictionary with evaluation metrics
    """
    start = time.time()
    y_pred_proba = model.predict_proba(X)[:, 1]
    inference_time = time.time() - start

    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y, y_pred_proba)
    acc = accuracy_score(y, y_pred)

    return {
        "model": name,
        "auc_roc": auc,
        "accuracy": acc,
        "inference_time_ms": inference_time * 1000,
        "samples": len(y),
    }
