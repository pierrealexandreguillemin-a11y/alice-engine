"""SHAP + Permutation Importance for trained models (ISO 25059).

Computes CatBoost native SHAP + manual permutation importance for all models.
Manual permutation avoids sklearn .fit() requirement (xgb.Booster/lgb.Booster crash).

Sources:
- Lundberg & Lee 2017 (SHAP): arXiv:1705.07874
- Breiman 2001 (permutation importance)
- CatBoost ShapValues: catboost.ai/docs/en/concepts/shap-values
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PERM_SUBSAMPLE = 20_000


def compute_shap_importance(
    results: dict,
    X_test: pd.DataFrame,
    y_test: Any,
    init_scores_test: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    """SHAP + manual permutation importance for trained models (ISO 25059).

    CatBoost: native SHAP (fast symmetric trees).
    LightGBM/XGBoost: shap.TreeExplainer (when available).
    All models: manual permutation importance.
    Returns concordance DataFrame saved to out_dir/feature_concordance.csv.
    """
    feature_names = list(X_test.columns)
    y_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
    n_models = sum(1 for r in results.values() if r.get("model") is not None)

    # 1. Model-specific SHAP (CatBoost native or TreeExplainer fallback)
    shap_df = _compute_model_shap(results, X_test, y_arr, init_scores_test, feature_names)
    shap_df.to_csv(out_dir / "catboost_shap_importance.csv", index=False)
    logger.info("SHAP top 15:\n%s", shap_df.head(15).to_string())

    # 2. Manual permutation importance — SKIP for single-model resume kernels
    # Skill: "Permutation: EXCLUDE from resume/continuation kernels. Use TreeSHAP + gain."
    # Budget: 197 features × 5 repeats × 17s = 4h39m with 85K trees — timeout guaranteed
    if n_models >= 2:
        perm = _permutation_all_models(results, X_test, y_arr, init_scores_test, feature_names)
        perm.to_csv(out_dir / "permutation_importance.csv", index=False)
    else:
        logger.info("Single-model mode — permutation skipped (use TreeSHAP + gain instead)")
        perm = pd.DataFrame({"feature": feature_names})

    # 3. Concordance matrix (meaningful only with 2+ models)
    concordance = _build_concordance(shap_df, perm, feature_names)
    concordance.to_csv(out_dir / "feature_concordance.csv", index=False)

    counts = concordance["verdict"].value_counts()
    logger.info("Feature concordance: %s", counts.to_dict())
    if n_models < 2:
        logger.info("Single-model mode — concordance VALIDATED requires 2+ models, skipped")
    n_val = (concordance["verdict"] == "VALIDATED").sum()
    logger.info(
        "VALIDATED features (%d):\n%s",
        n_val,
        concordance[concordance["verdict"] == "VALIDATED"][
            ["feature", "shap_importance", "mean_perm"]
        ]
        .head(30)
        .to_string(),
    )
    return concordance


def _compute_model_shap(
    results: dict,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    init_scores: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute SHAP for available model: CatBoost native or TreeExplainer fallback."""
    # Try CatBoost native SHAP first (fastest on symmetric trees)
    cb_model = results.get("CatBoost", {}).get("model")
    if cb_model is not None:
        return _catboost_native_shap(cb_model, X_test, y_test, init_scores, feature_names)

    # Fallback: shap.TreeExplainer for LightGBM/XGBoost
    for name in ("LightGBM", "XGBoost"):
        model = results.get(name, {}).get("model")
        if model is not None:
            return _tree_explainer_shap(model, name, X_test, feature_names)

    logger.warning("No model available for SHAP")
    return pd.DataFrame({"feature": feature_names, "shap_importance": 0.0})


def _catboost_native_shap(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    init_scores: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """CatBoost get_feature_importance(type='ShapValues') with auto axis detection."""
    from catboost import Pool  # noqa: PLC0415

    pool = Pool(X_test, label=y_test, baseline=init_scores)
    t0 = time.time()
    shap_vals = model.get_feature_importance(type="ShapValues", data=pool)
    logger.info("CatBoost SHAP: %.1fs, shape=%s", time.time() - t0, shap_vals.shape)

    n_feat = len(feature_names)
    if shap_vals.ndim == 3:
        if shap_vals.shape[1] == n_feat + 1:  # (samples, features+1, classes)
            mean_shap = np.abs(shap_vals[:, :n_feat, :]).mean(axis=0).sum(axis=1)
        elif shap_vals.shape[2] == n_feat + 1:  # (samples, classes, features+1)
            mean_shap = np.abs(shap_vals[:, :, :n_feat]).mean(axis=0).sum(axis=0)
        else:
            msg = f"Unexpected SHAP shape {shap_vals.shape} for {n_feat} features"
            raise ValueError(msg)
    else:
        mean_shap = np.abs(shap_vals[:, :n_feat]).mean(axis=0)

    df = pd.DataFrame({"feature": feature_names, "shap_importance": mean_shap})
    return df.sort_values("shap_importance", ascending=False).reset_index(drop=True)


def _tree_explainer_shap(
    model: Any,
    name: str,
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """shap.TreeExplainer for LightGBM/XGBoost (subsample for speed)."""
    import shap  # noqa: PLC0415

    n = min(PERM_SUBSAMPLE, len(X_test))
    rng = np.random.RandomState(42)  # noqa: NPY002
    idx = rng.choice(len(X_test), n, replace=False)
    X_sub = X_test.iloc[idx].reset_index(drop=True)
    # LGBMClassifier: use booster_ for TreeExplainer
    raw_model = getattr(model, "booster_", model)
    t0 = time.time()
    explainer = shap.TreeExplainer(raw_model)
    shap_vals = explainer.shap_values(X_sub)
    elapsed = time.time() - t0
    logger.info("%s TreeExplainer SHAP: %.1fs, type=%s", name, elapsed, type(shap_vals))

    # Multiclass: shap_vals is list of n_classes arrays (n_samples, n_features)
    if isinstance(shap_vals, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    elif shap_vals.ndim == 3:  # (n_samples, n_features, n_classes)
        mean_shap = np.abs(shap_vals).mean(axis=0).sum(axis=1)
    else:
        mean_shap = np.abs(shap_vals).mean(axis=0)

    df = pd.DataFrame({"feature": feature_names, "shap_importance": mean_shap})
    return df.sort_values("shap_importance", ascending=False).reset_index(drop=True)


def _permutation_all_models(
    results: dict,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    init_scores: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Manual permutation importance — no sklearn, no .fit() requirement."""
    from scripts.kaggle_metrics import predict_with_init  # noqa: PLC0415

    # Subsample for runtime (177 features × 5 repeats × 3 models on 221K = too slow)
    n = min(PERM_SUBSAMPLE, len(X_test))
    rng = np.random.RandomState(42)  # noqa: NPY002
    idx = rng.choice(len(X_test), n, replace=False)
    X_sub = X_test.iloc[idx].reset_index(drop=True)
    y_sub = y_test[idx]
    init_sub = init_scores[idx]
    logger.info("Permutation subsample: %d / %d", n, len(X_test))

    result = pd.DataFrame({"feature": feature_names})
    for name, r in results.items():
        if r["model"] is None:
            continue
        _model = r["model"]  # bind loop var for closure (B023)

        def _predict(X: pd.DataFrame, _m: Any = _model) -> np.ndarray:  # noqa: N803
            return predict_with_init(_m, X, init_sub)

        mean_imp, std_imp = _manual_permutation(
            _predict,
            X_sub,
            y_sub,
            feature_names,
            n_repeats=5,
            rng=rng,
        )
        result[f"{name}_perm_mean"] = mean_imp
        result[f"{name}_perm_std"] = std_imp
        logger.info("  %s permutation done", name)
        gc.collect()

    return result


def _manual_permutation(
    predict_fn: Callable,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    rng: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Permutation importance without sklearn (avoids Booster.fit() crash).

    Measures drop in neg-log-loss when each feature is shuffled.
    """
    from sklearn.metrics import log_loss  # noqa: PLC0415

    baseline = -log_loss(y, predict_fn(X), labels=[0, 1, 2])
    importances = np.zeros((len(feature_names), n_repeats))

    t0 = time.time()
    for i, col in enumerate(feature_names):
        original = X[col].values.copy()
        for r in range(n_repeats):
            X[col] = rng.permutation(original)
            score = -log_loss(y, predict_fn(X), labels=[0, 1, 2])
            importances[i, r] = baseline - score
        X[col] = original  # restore
    logger.info(
        "  Manual permutation: %.1fs (%d features × %d repeats)",
        time.time() - t0,
        len(feature_names),
        n_repeats,
    )
    return importances.mean(axis=1), importances.std(axis=1)


def _build_concordance(
    shap_df: pd.DataFrame,
    perm: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Merge SHAP + permutation into concordance matrix with verdict."""
    # Support both legacy 'catboost_shap' and new 'shap_importance' column names
    shap_col = "shap_importance" if "shap_importance" in shap_df.columns else "catboost_shap"
    concordance = (
        shap_df[["feature", shap_col]]
        .rename(columns={shap_col: "shap_importance"})
        .merge(perm, on="feature", how="left")
    )
    perm_cols = [c for c in concordance.columns if c.endswith("_perm_mean")]
    concordance["n_models_perm_positive"] = (concordance[perm_cols] > 0).sum(axis=1)
    concordance["mean_perm"] = concordance[perm_cols].mean(axis=1)
    concordance = concordance.sort_values("mean_perm", ascending=False).reset_index(drop=True)

    concordance["verdict"] = "NOISE"
    concordance.loc[concordance["n_models_perm_positive"] >= 1, "verdict"] = "MARGINAL"
    concordance.loc[concordance["n_models_perm_positive"] >= 2, "verdict"] = "VALIDATED"
    return concordance
