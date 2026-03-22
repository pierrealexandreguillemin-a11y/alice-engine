"""Multiclass metric helpers — ISO 25059/24029.

Document ID: ALICE-METRICS-MULTICLASS
Version: 1.1.0

Extracted from kaggle_diagnostics.py for ISO 5055 (<300 lines).
Provides: RPS, expected-score MAE, multiclass Brier, ECE, baseline conditions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_rps(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Ranked Probability Score — ordinal-aware metric (loss < draw < win).

    RPS penalises predictions far from the true outcome more heavily
    than those that are merely one class away.
    """
    n_classes = y_proba.shape[1]
    n = len(y_true)
    if n == 0:
        return 0.0
    cum_pred = np.cumsum(y_proba, axis=1)
    cum_true = np.zeros_like(cum_pred)
    for c in range(n_classes):
        cum_true[:, c] = (y_true <= c).astype(float)
    rps_per_row = np.mean((cum_pred - cum_true) ** 2, axis=1)
    return float(np.mean(rps_per_row))


def compute_expected_score_mae(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """MAE between predicted expected score and actual score.

    Expected score = 0*P(loss) + 0.5*P(draw) + 1*P(win).
    Actual score maps: class 0->0, class 1->0.5, class 2->1.
    """
    score_map = np.array([0.0, 0.5, 1.0])
    pred_score = y_proba @ score_map
    actual_score = score_map[y_true]
    return float(np.mean(np.abs(pred_score - actual_score)))


def compute_multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Multiclass Brier score: mean of per-class squared errors."""
    n_classes = y_proba.shape[1]
    one_hot = np.zeros_like(y_proba)
    for c in range(n_classes):
        one_hot[:, c] = (y_true == c).astype(float)
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def compute_ece(y_true: np.ndarray, y_proba_class: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for a single class column.

    Bins predicted probabilities, computes |mean_pred - empirical_freq|
    weighted by bin size.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_proba_class >= bin_edges[i]) & (y_proba_class < bin_edges[i + 1])
        if i == n_bins - 1:
            mask |= y_proba_class == bin_edges[i + 1]
        count = mask.sum()
        if count == 0:
            continue
        avg_pred = float(y_proba_class[mask].mean())
        avg_true = float(y_true[mask].mean())
        ece += (count / n) * abs(avg_pred - avg_true)
    return ece


def check_baseline_conditions(m: dict, baseline_metrics: dict) -> str | None:
    """Check 6 baseline conditions. Returns failure reason or None (ISO 25059).

    Conditions checked (model must beat both naive and Elo baselines):
    1. log_loss < naive  2. log_loss < elo  3. rps < naive
    4. rps < elo         5. brier < naive   6. es_mae < elo
    """
    naive = baseline_metrics.get("naive", {})
    elo = baseline_metrics.get("elo", {})
    checks: list[tuple[bool, str]] = [
        (
            bool(naive) and m.get("test_log_loss", 999) >= naive.get("log_loss", 999),
            "log_loss >= naive baseline",
        ),
        (
            bool(elo) and m.get("test_log_loss", 999) >= elo.get("log_loss", 999),
            "log_loss >= elo baseline",
        ),
        (bool(naive) and m.get("test_rps", 999) >= naive.get("rps", 999), "rps >= naive baseline"),
        (bool(elo) and m.get("test_rps", 999) >= elo.get("rps", 999), "rps >= elo baseline"),
        (
            bool(naive) and m.get("test_brier", 999) >= naive.get("brier", 999),
            "brier >= naive",
        ),
        (bool(elo) and m.get("test_es_mae", 999) >= elo.get("es_mae", 999), "es_mae >= elo"),
    ]
    for fail_cond, label in checks:
        if fail_cond:
            return label
    return None


def evaluate_on_test(results: dict, X_test: Any, y_test: Any) -> None:
    """Compute test metrics for each model (multiclass). Mutates results in-place."""
    from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

    for _name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_test)  # (n, 3)
        y_pred = np.argmax(y_proba, axis=1)
        y_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
        r["metrics"]["test_log_loss"] = float(log_loss(y_arr, y_proba))
        r["metrics"]["test_accuracy"] = float(accuracy_score(y_arr, y_pred))
        r["metrics"]["test_f1_macro"] = float(
            f1_score(y_arr, y_pred, average="macro", zero_division=0)
        )
        per_class_recall = recall_score(y_arr, y_pred, average=None, zero_division=0)
        r["metrics"]["recall_loss"] = (
            float(per_class_recall[0]) if len(per_class_recall) > 0 else 0.0
        )
        r["metrics"]["recall_draw"] = (
            float(per_class_recall[1]) if len(per_class_recall) > 1 else 0.0
        )
        r["metrics"]["recall_win"] = (
            float(per_class_recall[2]) if len(per_class_recall) > 2 else 0.0
        )
        r["metrics"]["test_rps"] = float(compute_rps(y_arr, y_proba))
        r["metrics"]["test_brier"] = float(compute_multiclass_brier(y_arr, y_proba))
        r["metrics"]["test_es_mae"] = float(compute_expected_score_mae(y_arr, y_proba))
