"""ML training diagnostics — ISO 42001/25059 compliance artifacts."""

from __future__ import annotations

import json  # noqa: E401
import logging
from pathlib import Path
from typing import Any

import numpy as np  # noqa: E401
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss, roc_curve

from scripts.kaggle_metrics import (
    _apply_isotonic,
    compute_ece,
    compute_expected_score_mae,
    compute_multiclass_brier,
    compute_rps,
    predict_with_init,
)

logger = logging.getLogger(__name__)


def save_diagnostics(
    results: dict,
    X_test: Any,
    y_test: Any,
    X_valid: Any,
    y_valid: Any,
    X_train: Any,
    out_dir: Path,
    init_scores_valid: Any = None,
    init_scores_test: Any = None,
    calibrators: dict | None = None,
) -> None:
    """Save all diagnostic artifacts for ISO 42001/25059/24029/5259 compliance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if calibrators is None:
        calibrators = calibrate_models(results, X_valid, y_valid, out_dir, init_scores_valid)
    _save_predictions(results, X_test, y_test, "test", out_dir, calibrators, init_scores_test)
    _save_predictions(results, X_valid, y_valid, "valid", out_dir, calibrators, init_scores_valid)
    _save_feature_importance(results, out_dir)
    _save_classification_reports(results, X_test, y_test, out_dir, init_scores_test)
    _save_learning_curves(results, out_dir)
    _save_roc_curves(results, X_test, y_test, out_dir, init_scores_test)
    _save_calibration_curves(results, X_test, y_test, out_dir, init_scores_test)
    _save_feature_distributions(X_train, out_dir)
    logger.info("Diagnostics saved to %s", out_dir)


def calibrate_models(
    results: dict,
    X_valid: Any,
    y_valid: Any,
    out_dir: Path,
    init_scores_valid: Any = None,
) -> dict:
    """Fit temperature scaling per model (Guo 2017, preserves E[score] — arXiv:2512.09054)."""
    import joblib  # noqa: PLC0415
    from scipy.optimize import minimize_scalar  # noqa: PLC0415

    calibrators: dict = {}
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = predict_with_init(r["model"], X_valid, init_scores_valid)
        y_true = y_valid.values if hasattr(y_valid, "values") else np.asarray(y_valid)
        logits = np.log(np.clip(y_proba, 1e-7, 1.0))
        idx = np.arange(len(y_true))

        def nll(T: float, _l: Any = logits, _i: Any = idx, _y: Any = y_true) -> float:
            s = _l / T
            s = s - s.max(axis=1, keepdims=True)
            p = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
            return -np.mean(np.log(np.clip(p[_i, _y], 1e-7, 1.0)))

        res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        calibrators[name] = float(res.x)
        logger.info("  %s temperature T=%.4f (NLL=%.4f)", name, res.x, res.fun)
    if calibrators:
        joblib.dump(calibrators, out_dir / "calibrators.joblib")
        logger.info("  Calibrators saved to %s", out_dir / "calibrators.joblib")
    return calibrators


def calibrate_models_dirichlet(
    results: dict,
    X_valid: Any,
    y_valid: Any,
    out_dir: Path,
    init_scores_valid: Any = None,
    lam: float = 1e-3,
) -> dict:
    """Fit Dirichlet calibration per model (Kull et al. 2019, NeurIPS).

    Dir-L2: full K×K affine on log-probas + L2 regularisation on off-diagonal.
    Captures inter-class interactions (draw↔loss/win) that temperature/isotonic miss.
    For K=3, fits 12 params (9 matrix + 3 bias) on 71K+ valid samples.
    """
    import joblib  # noqa: PLC0415
    from scipy.optimize import minimize  # noqa: PLC0415

    n_classes = 3
    calibrators: dict = {}
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = predict_with_init(r["model"], X_valid, init_scores_valid)
        y_true = y_valid.values if hasattr(y_valid, "values") else np.asarray(y_valid)
        logits = np.log(np.clip(y_proba, 1e-7, 1.0))
        n = len(y_true)
        idx = np.arange(n)

        def nll_l2(
            params: np.ndarray,
            _logits: Any = logits,
            _idx: Any = idx,
            _y: Any = y_true,
            _lam: float = lam,
        ) -> float:
            W = params[: n_classes * n_classes].reshape(n_classes, n_classes)
            b = params[n_classes * n_classes :]
            transformed = _logits @ W.T + b
            shifted = transformed - transformed.max(axis=1, keepdims=True)
            exp_t = np.exp(shifted)
            probs = exp_t / exp_t.sum(axis=1, keepdims=True)
            loss = -np.mean(np.log(np.clip(probs[_idx, _y], 1e-7, 1.0)))
            off_diag = W - np.diag(np.diag(W))
            return loss + _lam * np.sum(off_diag**2)

        x0 = np.zeros(n_classes * n_classes + n_classes)
        x0[: n_classes * n_classes] = np.eye(n_classes).ravel()
        res = minimize(nll_l2, x0, method="L-BFGS-B", options={"maxiter": 500})
        W = res.x[: n_classes * n_classes].reshape(n_classes, n_classes)
        b = res.x[n_classes * n_classes :]
        calibrators[name] = ("dirichlet", W, b)
        logger.info(
            "  %s Dirichlet: NLL=%.4f diag=[%.3f,%.3f,%.3f] off-diag L2=%.4f",
            name,
            res.fun,
            W[0, 0],
            W[1, 1],
            W[2, 2],
            np.sum((W - np.diag(np.diag(W))) ** 2),
        )
    if calibrators:
        joblib.dump(calibrators, out_dir / "calibrators_dirichlet.joblib")
    return calibrators


def calibrate_models_isotonic(
    results: dict,
    X_valid: Any,
    y_valid: Any,
    out_dir: Path,
    init_scores_valid: Any = None,
) -> dict:
    """Fit isotonic regression per class (Niculescu-Mizil & Caruana 2005)."""
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

    calibrators: dict = {}
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = predict_with_init(r["model"], X_valid, init_scores_valid)
        y_true = y_valid.values if hasattr(y_valid, "values") else np.asarray(y_valid)
        isos = [
            IsotonicRegression(out_of_bounds="clip").fit(y_proba[:, c], (y_true == c).astype(float))
            for c in range(3)
        ]
        calibrators[name] = isos
        logger.info("  %s isotonic fitted (3 classes)", name)
    return calibrators


def _save_predictions(
    results: dict,
    X: Any,
    y: Any,
    split: str,
    out_dir: Path,
    calibrators: dict | None = None,
    init_scores: Any = None,
) -> None:
    """Save per-model multiclass predictions as parquet (raw + calibrated)."""
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = predict_with_init(r["model"], X, init_scores)
        y_pred = np.argmax(y_proba, axis=1)
        data: dict = {
            "y_true": y.values if hasattr(y, "values") else y,
            "y_proba_loss": y_proba[:, 0],
            "y_proba_draw": y_proba[:, 1],
            "y_proba_win": y_proba[:, 2],
            "y_pred": y_pred,
        }
        if calibrators and name in calibrators:
            y_cal = _apply_isotonic(y_proba, calibrators[name])
            data["y_proba_cal_loss"] = y_cal[:, 0]
            data["y_proba_cal_draw"] = y_cal[:, 1]
            data["y_proba_cal_win"] = y_cal[:, 2]
            data["y_pred_calibrated"] = np.argmax(y_cal, axis=1)
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
    init_scores: Any = None,
) -> None:
    """Save 3-class classification report per model (ISO 25059)."""
    reports = {}
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = predict_with_init(r["model"], X_test, init_scores)
        y_pred = np.argmax(y_proba, axis=1)
        reports[name] = classification_report(
            y_test,
            y_pred,
            target_names=["loss", "draw", "win"],
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
        evals = getattr(model, "evals_result_", None) or {}
        if "val" in evals:
            metrics = evals["val"]
            key = next(iter(metrics))
            return pd.DataFrame({"iteration": range(len(metrics[key])), key: metrics[key]})
    if name == "LightGBM" and hasattr(model, "evals_result_"):
        evals = model.evals_result_
        if "valid_0" in evals:
            metrics = evals["valid_0"]
            key = next(iter(metrics))
            return pd.DataFrame({"iteration": range(len(metrics[key])), key: metrics[key]})
    return None


def _save_roc_curves(
    results: dict,
    X_test: Any,
    y_test: Any,
    out_dir: Path,
    init_scores: Any = None,
) -> None:
    """Save per-class ROC curve data (one-vs-rest) per model (ISO 25059)."""
    class_names = ["loss", "draw", "win"]
    count = 0
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = predict_with_init(r["model"], X_test, init_scores)
        y_arr = np.asarray(y_test)
        for c, cname in enumerate(class_names):
            binary_true = (y_arr == c).astype(int)
            fpr, tpr, thresholds = roc_curve(binary_true, y_proba[:, c])
            df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
            df.to_csv(out_dir / f"{name}_roc_{cname}.csv", index=False)
        count += 1
    logger.info("  ROC curves saved (%d models, 3 classes each)", count)


def _save_calibration_curves(
    results: dict,
    X_test: Any,
    y_test: Any,
    out_dir: Path,
    init_scores: Any = None,
) -> None:
    """Save per-class calibration curves per model (ISO 24029 robustness)."""
    class_names = ["loss", "draw", "win"]
    for name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = predict_with_init(r["model"], X_test, init_scores)
        y_arr = np.asarray(y_test)
        for c, cname in enumerate(class_names):
            binary_true = (y_arr == c).astype(int)
            prob_true, prob_pred = calibration_curve(
                binary_true,
                y_proba[:, c],
                n_bins=10,
                strategy="uniform",
            )
            pd.DataFrame({"mean_predicted": prob_pred, "fraction_positive": prob_true}).to_csv(
                out_dir / f"{name}_calibration_{cname}.csv",
                index=False,
            )
    logger.info("  Calibration curves saved (3 classes per model)")


def _save_feature_distributions(X_train: Any, out_dir: Path) -> None:
    """Save train feature distributions as drift baseline (ISO 5259 data quality)."""
    if X_train is None or not hasattr(X_train, "describe"):
        return
    stats = X_train.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats.to_csv(out_dir / "train_feature_distributions.csv")
    logger.info("  Feature distributions saved (%d features)", len(stats))


def _compute_metrics(y_true: Any, y_pred: Any, y_proba: Any) -> dict:
    """Multiclass (3-class) validation metrics (ISO 25059)."""
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)  # (n, 3)

    # Naive baseline: always predict marginal distribution
    counts = np.bincount(y_true_arr, minlength=3)
    marginal = counts / counts.sum()
    baseline_proba = np.tile(marginal, (len(y_true_arr), 1))
    baseline_ll = float(log_loss(y_true_arr, baseline_proba))
    model_ll = float(log_loss(y_true_arr, y_proba_arr))

    observed_draw = float((y_true_arr == 1).mean())
    mean_p_draw = float(y_proba_arr[:, 1].mean())

    class_names = ["loss", "draw", "win"]
    ece_dict = {}
    for c, name in enumerate(class_names):
        binary_true = (y_true_arr == c).astype(float)
        ece_dict[f"ece_class_{name}"] = compute_ece(binary_true, y_proba_arr[:, c])

    # fmt: off
    return {
        "log_loss": model_ll,
        "log_loss_baseline": baseline_ll,
        "log_loss_ratio": round(model_ll / baseline_ll, 4) if baseline_ll > 0 else 0.0,
        "rps": compute_rps(y_true_arr, y_proba_arr),
        "expected_score_mae": compute_expected_score_mae(y_true_arr, y_proba_arr),
        "brier_multiclass": compute_multiclass_brier(y_true_arr, y_proba_arr),
        **ece_dict,
        "mean_p_draw": mean_p_draw,
        "observed_draw_rate": observed_draw,
        "draw_calibration_bias": round(mean_p_draw - observed_draw, 6),
        "accuracy_3class": float(accuracy_score(y_true_arr, y_pred)),
        "f1_macro": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
    }
    # fmt: on
