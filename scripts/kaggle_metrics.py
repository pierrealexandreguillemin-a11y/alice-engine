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


class XGBWrapper:
    """Wrap xgb.Booster for sklearn-compatible pipeline (predict_proba, save_model)."""

    def __init__(self, booster: Any, feature_names: Any, n_classes: int = 3) -> None:
        """Init wrapper with Booster, feature names, and class count."""
        self._booster = booster
        self._n_classes = n_classes
        scores = booster.get_score(importance_type="gain")
        self.feature_importances_ = np.array([scores.get(f, 0.0) for f in feature_names])

    def predict_proba(self, X: Any) -> Any:
        """Return (n, n_classes) probability matrix via DMatrix predict."""
        import xgboost as xgb  # noqa: PLC0415, F811

        return self._booster.predict(xgb.DMatrix(X)).reshape(-1, self._n_classes)

    def save_model(self, path: str) -> None:
        """Delegate to Booster.save_model."""
        self._booster.save_model(path)

    def get_booster(self) -> Any:
        """Return underlying xgb.Booster for DMatrix predictions with base_margin."""
        return self._booster


def predict_with_init(
    model: Any,
    X: Any,
    init_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Predict probas with init_scores for residual learning (per-library API).

    CatBoost: Pool(baseline=). XGBoost: DMatrix.set_base_margin. LightGBM: raw + manual softmax.
    """
    cls = type(model).__name__
    if init_scores is None:
        if cls == "Booster":
            import xgboost as xgb  # noqa: PLC0415

            return np.asarray(model.predict(xgb.DMatrix(X))).reshape(-1, 3)
        return np.asarray(model.predict_proba(X))  # works for _XGBWrapper too
    if cls == "CatBoostClassifier":
        from catboost import Pool  # noqa: PLC0415

        return np.asarray(model.predict_proba(Pool(X, baseline=init_scores)))
    if cls in ("XGBClassifier", "Booster", "XGBWrapper"):
        import xgboost as xgb  # noqa: PLC0415

        dm = xgb.DMatrix(X)
        dm.set_base_margin(init_scores.ravel())
        bst = model.get_booster() if hasattr(model, "get_booster") else model
        return np.asarray(bst.predict(dm)).reshape(-1, init_scores.shape[1])
    if cls == "LGBMClassifier":
        raw = np.asarray(model.predict(X, raw_score=True))
        adjusted = raw + init_scores
        exp_s = np.exp(adjusted - adjusted.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)
    return np.asarray(model.predict_proba(X))


def evaluate_on_test(
    results: dict,
    X_test: Any,
    y_test: Any,
    init_scores_test: Any | None = None,
) -> None:
    """Compute test metrics for each model (multiclass). Mutates results in-place."""
    from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

    for _name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = predict_with_init(r["model"], X_test, init_scores_test)
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
        r["metrics"]["mean_p_draw"] = float(y_proba[:, 1].mean()) if y_proba.shape[1] > 1 else 0.0


def _check_calibration(m: dict) -> str | None:
    """Check ECE per class, draw bias, and draw recall (conditions 7-9)."""
    for cls in ("loss", "draw", "win"):
        if m.get(f"ece_class_{cls}", 1.0) >= 0.05:
            return f"ece_{cls} >= 0.05"
    if abs(m.get("draw_calibration_bias", 1.0)) >= 0.02:
        return "draw_calibration_bias >= 0.02"
    if m.get("mean_p_draw", 0.0) < 0.01:
        return "mean_p_draw < 0.01 (model produces near-zero draw probabilities)"
    return None


def check_quality_gates(
    results: dict,
    baseline_metrics: dict | None = None,
    champion_ll: float | None = None,
) -> dict:
    """ISO 42001: 9-condition quality gate (baselines + calibration + draw recall + champion)."""
    candidates = [(n, r) for n, r in results.items() if r["model"] is not None]
    if not candidates:
        return {"passed": False, "reason": "All models failed to train"}
    best_name, best_r = min(candidates, key=lambda x: x[1]["metrics"].get("test_log_loss", 999.0))
    m = best_r["metrics"]
    best_ll = m.get("test_log_loss", 999.0)
    if baseline_metrics:
        reason = check_baseline_conditions(m, baseline_metrics)
        if reason:
            return {"passed": False, "reason": reason}
    # Calibration + draw prediction checks (conditions 7-9)
    cal_reason = _check_calibration(m)
    if cal_reason:
        return {"passed": False, "reason": cal_reason}
    if champion_ll and champion_ll > 0:
        rise_pct = (best_ll - champion_ll) / champion_ll * 100
        if rise_pct > 5.0:
            return {"passed": False, "reason": f"Degradation {rise_pct:.1f}% > 5.0%"}
    return {"passed": True, "best_model": best_name, "best_log_loss": best_ll}
