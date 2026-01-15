"""Module: metrics.py - Métriques Stacking Ensemble.

Extrait de stacking.py pour conformité ISO 5055 (<300 lignes).

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)
- ISO/IEC 42001:2023 - AI Management System

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scripts.ensemble.voting import MODEL_NAMES

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scripts.ensemble.types import StackingMetrics
    from scripts.ml_types import MLClassifier


def extract_model_weights(meta_model: MLClassifier) -> dict[str, float]:
    """Extract model weights from meta-learner."""
    if hasattr(meta_model, "coef_"):
        coefs = meta_model.coef_[0]
        weights = np.abs(coefs) / np.abs(coefs).sum()
        return {name: float(w) for name, w in zip(MODEL_NAMES, weights, strict=False)}
    return {name: 1.0 / len(MODEL_NAMES) for name in MODEL_NAMES}


def compute_stacking_metrics(
    model_aucs: dict[str, float],
    stacking_train_auc: float,
    stacking_test_auc: float,
    soft_voting_auc: float,
    y_test_np: NDArray[np.int64],
    test_matrix: NDArray[np.float64],
) -> StackingMetrics:
    """Compute final stacking metrics."""
    from sklearn.metrics import roc_auc_score

    from scripts.ensemble.types import StackingMetrics

    single_models: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(MODEL_NAMES):
        test_auc = float(roc_auc_score(y_test_np, test_matrix[:, idx]))
        single_models[name] = {"oof_auc": model_aucs[name], "test_auc": test_auc}

    best_name = max(single_models, key=lambda x: single_models[x]["test_auc"])
    best_auc = single_models[best_name]["test_auc"]

    return StackingMetrics(
        single_models=single_models,
        stacking_train_auc=stacking_train_auc,
        stacking_test_auc=stacking_test_auc,
        soft_voting_test_auc=soft_voting_auc,
        gain_vs_best_single=stacking_test_auc - best_auc,
        gain_vs_soft_voting=stacking_test_auc - soft_voting_auc,
        best_single_name=best_name,
        best_single_auc=best_auc,
    )
