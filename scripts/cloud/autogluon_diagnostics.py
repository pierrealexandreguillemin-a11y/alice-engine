"""AutoGluon diagnostics — ISO 24029/24027/42001 (ISO 5055 SRP split)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_predictions(predictor: Any, test_data: pd.DataFrame, label: str, out_dir: Path) -> None:
    """Save test predictions for future McNemar comparison."""
    y_proba = predictor.predict_proba(test_data.drop(columns=label))[1]
    preds = pd.DataFrame(
        {
            "y_true": test_data[label].values,
            "y_proba": y_proba.values,
            "y_pred": (y_proba >= 0.5).astype(int).values,
        }
    )
    preds.to_parquet(out_dir / "predictions_test.parquet", index=False)
    logger.info("Predictions saved: %d rows", len(preds))


def save_diagnostics(
    predictor: Any,
    test_data: pd.DataFrame,
    label: str,
    leaderboard: Any,
    out_dir: Path,
) -> None:
    """ROC, calibration, feature importance, leaderboard."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415, E402
    from sklearn.calibration import calibration_curve  # noqa: PLC0415
    from sklearn.metrics import RocCurveDisplay  # noqa: PLC0415

    y_true = test_data[label]
    y_proba = predictor.predict_proba(test_data.drop(columns=label))[1]

    # ROC curve
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close()

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, "s-", label="AutoGluon")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig(out_dir / "calibration_curve.png", dpi=150)
    plt.close()

    # Feature importance
    importance = predictor.feature_importance(test_data, silent=True)
    importance.to_csv(out_dir / "feature_importance.csv")

    # Leaderboard
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    logger.info("Diagnostics saved: ROC, calibration, importance, leaderboard")


def test_robustness(
    predictor: Any,
    test_data: pd.DataFrame,
    label: str,
    noise_level: float = 0.1,
    seed: int = 42,
) -> dict:
    """ISO 24029: Gaussian noise perturbation test."""
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    X_test = test_data.drop(columns=label)
    y_true = test_data[label]
    baseline_auc = float(roc_auc_score(y_true, predictor.predict_proba(X_test)[1]))

    X_noisy = X_test.copy()
    rng = np.random.default_rng(seed)
    for col in X_noisy.select_dtypes(include=[np.number]).columns:
        std = X_noisy[col].std()
        if std > 0:
            X_noisy[col] = X_noisy[col] + rng.normal(0, std * noise_level, len(X_noisy))
    noisy_auc = float(roc_auc_score(y_true, predictor.predict_proba(X_noisy)[1]))

    tolerance = noisy_auc / baseline_auc if baseline_auc > 0 else 0
    status = "ROBUST" if tolerance >= 0.95 else ("ACCEPTABLE" if tolerance >= 0.90 else "FRAGILE")
    return {
        "baseline_auc": baseline_auc,
        "noisy_auc": noisy_auc,
        "noise_level": noise_level,
        "tolerance": tolerance,
        "status": status,
    }


def test_fairness(
    predictor: Any,
    test_data: pd.DataFrame,
    label: str,
    attr: str = "ligue_code",
) -> dict:
    """ISO 24027: Demographic parity across groups."""
    X_test = test_data.drop(columns=label)
    y_pred = predictor.predict(X_test)
    groups = test_data[attr].value_counts()
    rates = {}
    for group in groups[groups >= 100].index:
        mask = test_data[attr] == group
        rates[str(group)] = float(y_pred[mask].mean())
    vals = list(rates.values())
    parity = max(vals) - min(vals) if vals else 0
    status = "FAIR" if parity < 0.05 else ("ACCEPTABLE" if parity < 0.10 else "CRITICAL")
    return {
        "attribute": attr,
        "positive_rates": rates,
        "demographic_parity": parity,
        "status": status,
    }
