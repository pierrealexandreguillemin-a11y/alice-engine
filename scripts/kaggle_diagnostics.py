"""ML training diagnostics — ISO 42001/25059 compliance artifacts.

Saves predictions, feature importance, learning curves, and
classification reports alongside trained models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def save_diagnostics(
    results: dict,
    X_test: Any,
    y_test: Any,
    X_valid: Any,
    y_valid: Any,
    X_train: Any,
    out_dir: Path,
) -> None:
    """Save all diagnostic artifacts for ISO 42001/25059/24029/5259 compliance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    calibrators = calibrate_models(results, X_valid, y_valid, out_dir)
    _save_predictions(results, X_test, y_test, "test", out_dir, calibrators)
    _save_predictions(results, X_valid, y_valid, "valid", out_dir, calibrators)
    _save_feature_importance(results, out_dir)
    _save_classification_reports(results, X_test, y_test, out_dir)
    _save_learning_curves(results, out_dir)
    _save_roc_curves(results, X_test, y_test, out_dir)
    _save_calibration_curves(results, X_test, y_test, out_dir)
    _save_feature_distributions(X_train, out_dir)
    logger.info("Diagnostics saved to %s", out_dir)


def calibrate_models(
    results: dict,
    X_valid: Any,
    y_valid: Any,
    out_dir: Path,
) -> dict:
    """Fit isotonic calibration per model on validation set. Save calibrators."""
    import joblib  # noqa: PLC0415
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

    calibrators: dict = {}
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_valid)[:, 1]
        y_true = y_valid.values if hasattr(y_valid, "values") else y_valid
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(y_proba, y_true)
        calibrators[name] = iso
        logger.info("  %s isotonic calibration fitted (valid set)", name)
    if calibrators:
        joblib.dump(calibrators, out_dir / "calibrators.joblib")
        logger.info("  Calibrators saved to %s", out_dir / "calibrators.joblib")
    return calibrators


def _save_predictions(
    results: dict,
    X: Any,
    y: Any,
    split: str,
    out_dir: Path,
    calibrators: dict | None = None,
) -> None:
    """Save per-model predictions as parquet (raw + calibrated)."""
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        data: dict = {
            "y_true": y.values if hasattr(y, "values") else y,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }
        if calibrators and name in calibrators:
            y_cal = calibrators[name].predict(y_proba)
            data["y_proba_calibrated"] = y_cal
            data["y_pred_calibrated"] = (y_cal >= 0.5).astype(int)
        path = out_dir / f"{name}_{split}_predictions.parquet"
        pd.DataFrame(data).to_parquet(path, index=False)
    logger.info(
        "  Predictions saved for %s split (%d models)",
        split,
        sum(1 for r in results.values() if r["model"] is not None),
    )


def _save_feature_importance(results: dict, out_dir: Path) -> None:
    """Save feature importance as CSV per model (ISO 25059 explicability)."""
    for name, r in results.items():
        if not r.get("importance"):
            continue
        imp = sorted(r["importance"].items(), key=lambda x: -abs(x[1]))
        df = pd.DataFrame(imp, columns=["feature", "importance"])
        df.to_csv(out_dir / f"{name}_feature_importance.csv", index=False)
    logger.info("  Feature importance saved")


def _save_classification_reports(
    results: dict,
    X_test: Any,
    y_test: Any,
    out_dir: Path,
) -> None:
    """Save classification report per model (ISO 25059)."""
    from sklearn.metrics import classification_report  # noqa: PLC0415

    reports = {}
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        reports[name] = classification_report(
            y_test,
            y_pred,
            target_names=["loss/draw", "win"],
            output_dict=True,
        )
    with open(out_dir / "classification_reports.json", "w") as f:
        json.dump(reports, f, indent=2, default=str)
    logger.info("  Classification reports saved")


def _save_learning_curves(results: dict, out_dir: Path) -> None:
    """Save eval metrics per iteration for XGBoost/LightGBM (ISO 42001)."""
    for name, r in results.items():
        model = r.get("model")
        if model is None:
            continue
        curve = _extract_curve(name, model)
        if curve is not None:
            curve.to_csv(out_dir / f"{name}_learning_curve.csv", index=False)
    logger.info("  Learning curves saved")


def _extract_curve(name: str, model: Any) -> pd.DataFrame | None:
    """Extract eval history from a trained model."""
    if name == "CatBoost" and hasattr(model, "get_evals_result"):
        evals = model.get_evals_result()
        for _set_name, metrics in evals.items():
            if metrics:
                key = next(iter(metrics))
                return pd.DataFrame({"iteration": range(len(metrics[key])), key: metrics[key]})
    if name == "XGBoost":
        evals = model.evals_result() if callable(getattr(model, "evals_result", None)) else {}
        if "validation_0" in evals:
            metrics = evals["validation_0"]
            key = next(iter(metrics))
            return pd.DataFrame({"iteration": range(len(metrics[key])), key: metrics[key]})
    if name == "LightGBM" and hasattr(model, "evals_result_"):
        evals = model.evals_result_
        if "valid_0" in evals:
            metrics = evals["valid_0"]
            key = next(iter(metrics))
            return pd.DataFrame({"iteration": range(len(metrics[key])), key: metrics[key]})
    return None


def _save_roc_curves(results: dict, X_test: Any, y_test: Any, out_dir: Path) -> None:
    """Save ROC curve data points per model (ISO 25059 explicability)."""
    from sklearn.metrics import roc_curve  # noqa: PLC0415

    roc_data = {}
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_data[name] = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
        roc_data[name].to_csv(out_dir / f"{name}_roc_curve.csv", index=False)
    logger.info("  ROC curves saved (%d models)", len(roc_data))


def _save_calibration_curves(results: dict, X_test: Any, y_test: Any, out_dir: Path) -> None:
    """Save calibration curve data per model (ISO 24029 robustness)."""
    from sklearn.calibration import calibration_curve  # noqa: PLC0415

    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")
        pd.DataFrame({"mean_predicted": prob_pred, "fraction_positive": prob_true}).to_csv(
            out_dir / f"{name}_calibration_curve.csv", index=False
        )
    logger.info("  Calibration curves saved")


def _save_feature_distributions(X_train: Any, out_dir: Path) -> None:
    """Save train feature distributions as drift baseline (ISO 5259 data quality)."""
    if X_train is None or not hasattr(X_train, "describe"):
        return
    stats = X_train.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats.to_csv(out_dir / "train_feature_distributions.csv")
    logger.info("  Feature distributions saved (%d features)", len(stats))


def _compute_metrics(y_true: Any, y_pred: Any, y_proba: Any) -> dict:
    """Validation metrics with calibration quality (ISO 25059)."""
    import numpy as np  # noqa: PLC0415
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    cm = confusion_matrix(y_true, y_pred)
    pos_rate = float(np.mean(y_true))
    baseline_ll = float(log_loss(y_true, np.full(len(y_true), pos_rate)))
    model_ll = float(log_loss(y_true, y_proba))
    # fmt: off
    return {"auc_roc": float(roc_auc_score(y_true, y_proba)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "log_loss": model_ll,
            "log_loss_baseline": baseline_ll,
            "log_loss_ratio": round(model_ll / baseline_ll, 4) if baseline_ll > 0 else 0.0,
            "brier_score": float(brier_score_loss(y_true, y_proba)),
            "mean_proba": float(np.mean(y_proba)),
            "positive_rate": pos_rate,
            "true_negatives": int(cm[0, 0]), "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]), "true_positives": int(cm[1, 1])}
    # fmt: on
