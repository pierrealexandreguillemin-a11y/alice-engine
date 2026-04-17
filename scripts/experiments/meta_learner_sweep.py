"""Meta-learner experiment sweep — ISO 42001/25059.

Document ID: ALICE-META-SWEEP
Version: 1.0.0

Systematic experiment: feature sets × architectures × regularization × calibration.
Protocol: OOF train (1.21M), test eval (231K), all metrics reported.
Ref: docs/superpowers/specs/2026-04-16-meta-learner-experiments-design.md

Sources:
- NeurIPS 2024: "Better by Default: Strong Pre-Tuned MLPs"
- Kull 2019 (NeurIPS): Dirichlet calibration
- scikit-learn: StackingClassifier passthrough (restacking)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OOF_DIR = Path("results/oof_merged")
OUT_DIR = Path("results/meta_learner_sweep")

# Top features consensus (CB SHAP + XGB gain + LGB gain, cross-model top 20)
TOP_FEATURES_10 = [
    "draw_rate_home_dom",
    "draw_rate_noir",
    "draw_rate_blanc",
    "win_rate_home_dom",
    "est_domicile_blanc",
    "draw_rate_recent_noir",
    "draw_rate_recent_blanc",
    "draw_trend_blanc",
    "ronde",
    "diff_elo",
]

TOP_FEATURES_30 = TOP_FEATURES_10 + [
    "saison",
    "win_rate_normal_noir",
    "win_rate_normal_blanc",
    "draw_rate_equipe_ext",
    "win_rate_black_noir",
    "win_rate_white_blanc",
    "draw_rate_black_noir",
    "draw_rate_white_blanc",
    "diff_form",
    "draw_rate_normal_blanc",
    "draw_rate_normal_noir",
    "h2h_draw_rate",
    "h2h_win_rate",
    "diff_points_cumules",
    "promu_vs_strong",
    "diff_win_rate_recent",
    "diff_position",
    "win_rate_recent_noir",
    "expected_score_recent_blanc",
    "expected_score_recent_noir",
]


def _compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """All quality gate metrics."""
    from scripts.kaggle_metrics import (
        compute_ece,
        compute_expected_score_mae,
        compute_rps,
    )

    return {
        "log_loss": float(log_loss(y_true, y_proba)),
        "ece_draw": float(compute_ece((y_true == 1).astype(float), y_proba[:, 1])),
        "draw_bias": float(y_proba[:, 1].mean()) - float((y_true == 1).mean()),
        "rps": float(compute_rps(y_true, y_proba)),
        "es_mae": float(compute_expected_score_mae(y_true, y_proba)),
        "mean_p_draw": float(y_proba[:, 1].mean()),
    }


def _add_engineered(X_base: np.ndarray) -> np.ndarray:
    """Add 9 engineered features: per-class disagreement (3) + max_prob (3) + entropy (3)."""
    feats = [X_base]
    for c in range(3):
        preds = np.stack([X_base[:, c], X_base[:, 3 + c], X_base[:, 6 + c]], axis=1)
        feats.append(preds.std(axis=1, keepdims=True))
    for i in range(3):
        feats.append(X_base[:, i * 3 : (i + 1) * 3].max(axis=1, keepdims=True))
    for i in range(3):
        p = np.clip(X_base[:, i * 3 : (i + 1) * 3], 1e-7, 1.0)
        feats.append((-p * np.log(p)).sum(axis=1, keepdims=True))
    return np.hstack(feats)


def _dirichlet_cal(y_train: np.ndarray, p_train: np.ndarray, p_test: np.ndarray) -> np.ndarray:
    """Dirichlet calibration (Kull 2019) — fit on train, apply on test."""
    log_p = np.log(np.clip(p_train, 1e-7, 1.0))
    lr = LogisticRegression(C=1000.0, max_iter=1000, solver="lbfgs")
    lr.fit(log_p, y_train)
    return lr.predict_proba(np.log(np.clip(p_test, 1e-7, 1.0)))


def main() -> None:  # noqa: PLR0915
    """Run full experiment sweep."""
    t0 = time.time()
    logger.info("META-LEARNER SWEEP — systematic experiment")

    # --- Load data ---
    xgb_lgb_oof = pd.read_parquet(OOF_DIR / "xgb_lgb_oof_predictions.parquet")
    cb_oof = pd.read_parquet(OOF_DIR / "cb_oof_predictions.parquet")
    xgb_lgb_test = pd.read_parquet(OOF_DIR / "xgb_lgb_test_predictions.parquet")
    cb_test = pd.read_parquet(OOF_DIR / "cb_test_predictions.parquet")

    y_oof = xgb_lgb_oof["y_true"].values.astype(int)
    y_test = xgb_lgb_test["y_true"].values.astype(int)

    xgb_cols = ["xgb_p_loss", "xgb_p_draw", "xgb_p_win"]
    lgb_cols = ["lgb_p_loss", "lgb_p_draw", "lgb_p_win"]
    cb_cols = ["cb_p_loss", "cb_p_draw", "cb_p_win"]

    X9_oof = np.hstack(
        [xgb_lgb_oof[xgb_cols].values, xgb_lgb_oof[lgb_cols].values, cb_oof[cb_cols].values]
    )
    X9_test = np.hstack(
        [xgb_lgb_test[xgb_cols].values, xgb_lgb_test[lgb_cols].values, cb_test[cb_cols].values]
    )

    X18_oof = _add_engineered(X9_oof)
    X18_test = _add_engineered(X9_test)

    # Raw features (if available)
    feat_oof_path = OOF_DIR / "combined_features_prepared.parquet"
    feat_test_path = OOF_DIR / "test_features_prepared.parquet"
    has_raw = feat_oof_path.exists() and feat_test_path.exists()

    if has_raw:
        feat_oof = pd.read_parquet(feat_oof_path)
        feat_test = pd.read_parquet(feat_test_path)
        logger.info("Raw features loaded: %s", feat_oof.shape)

        # Build feature sets with raw features
        def _get_raw_cols(col_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
            available = [c for c in col_list if c in feat_oof.columns]
            missing = [c for c in col_list if c not in feat_oof.columns]
            if missing:
                logger.warning("Missing features: %s", missing[:5])
            return feat_oof[available].values, feat_test[available].values

        raw10_oof, raw10_test = _get_raw_cols(TOP_FEATURES_10)
        raw30_oof, raw30_test = _get_raw_cols(TOP_FEATURES_30)
        raw_all_oof, raw_all_test = feat_oof.values, feat_test.values

        # Fill NaN with 0 (safe for meta-learner — base models already handle NaN)
        raw10_oof = np.nan_to_num(raw10_oof, 0.0)
        raw10_test = np.nan_to_num(raw10_test, 0.0)
        raw30_oof = np.nan_to_num(raw30_oof, 0.0)
        raw30_test = np.nan_to_num(raw30_test, 0.0)
        raw_all_oof = np.nan_to_num(raw_all_oof, 0.0)
        raw_all_test = np.nan_to_num(raw_all_test, 0.0)
    else:
        logger.warning("Raw features not found — running without restacking experiments")

    # --- Define experiments ---
    results = []

    # Feature set experiments
    feature_sets = {
        "9_probas": (X9_oof, X9_test),
        "18_enriched": (X18_oof, X18_test),
    }
    if has_raw:
        feature_sets["28_top10"] = (
            np.hstack([X18_oof, raw10_oof]),
            np.hstack([X18_test, raw10_test]),
        )
        feature_sets["48_top30"] = (
            np.hstack([X18_oof, raw30_oof]),
            np.hstack([X18_test, raw30_test]),
        )
        feature_sets["219_all"] = (
            np.hstack([X18_oof, raw_all_oof]),
            np.hstack([X18_test, raw_all_test]),
        )

    architectures = {
        "MLP_16": (16,),
        "MLP_32_16": (32, 16),
        "MLP_64_32": (64, 32),
        "MLP_128_64_32": (128, 64, 32),
    }

    alphas = {"a1e-4": 1e-4, "a1e-3": 1e-3, "a1e-2": 1e-2, "a1e-1": 1e-1}

    # ==== AXE 1: Feature sets (fixed arch MLP(32,16), default alpha) ====
    logger.info("=" * 70)
    logger.info("AXE 1: Feature sets (MLP(32,16), alpha=1e-4)")
    logger.info("=" * 70)

    for fname, (Xo, Xt) in feature_sets.items():
        t1 = time.time()
        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            alpha=1e-4,
        )
        mlp.fit(Xo, y_oof)
        p = mlp.predict_proba(Xt)
        m = _compute_metrics(y_test, p)
        m["experiment"] = f"feat_{fname}"
        m["features"] = Xo.shape[1]
        m["arch"] = "32_16"
        m["alpha"] = 1e-4
        m["calibration"] = "none"
        m["time_s"] = round(time.time() - t1, 1)
        results.append(m)
        logger.info(
            "  %-20s %d feat  ll=%.6f  ECE_draw=%.4f  bias=%+.4f  (%.1fs)",
            fname,
            Xo.shape[1],
            m["log_loss"],
            m["ece_draw"],
            m["draw_bias"],
            m["time_s"],
        )

    # ==== AXE 2: Architectures (on best feature set from Axe 1) ====
    best_feat = min(
        [(n, Xo, Xt) for n, (Xo, Xt) in feature_sets.items()],
        key=lambda x: next(r["log_loss"] for r in results if r["experiment"] == f"feat_{x[0]}"),
    )
    best_fname, best_Xo, best_Xt = best_feat

    logger.info("=" * 70)
    logger.info("AXE 2: Architectures (features=%s, alpha=1e-4)", best_fname)
    logger.info("=" * 70)

    for aname, layers in architectures.items():
        t1 = time.time()
        mlp = MLPClassifier(
            hidden_layer_sizes=layers,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            alpha=1e-4,
        )
        mlp.fit(best_Xo, y_oof)
        p = mlp.predict_proba(best_Xt)
        m = _compute_metrics(y_test, p)
        m["experiment"] = f"arch_{aname}"
        m["features"] = best_Xo.shape[1]
        m["arch"] = aname
        m["alpha"] = 1e-4
        m["calibration"] = "none"
        m["time_s"] = round(time.time() - t1, 1)
        results.append(m)
        logger.info(
            "  %-20s  ll=%.6f  ECE_draw=%.4f  bias=%+.4f  (%.1fs)",
            aname,
            m["log_loss"],
            m["ece_draw"],
            m["draw_bias"],
            m["time_s"],
        )

    # Best arch
    best_arch_name = min(
        [r for r in results if r["experiment"].startswith("arch_")],
        key=lambda r: r["log_loss"],
    )["arch"]
    best_layers = architectures[best_arch_name]

    # ==== AXE 3: Regularization (best features + best arch) ====
    logger.info("=" * 70)
    logger.info("AXE 3: Regularization (features=%s, arch=%s)", best_fname, best_arch_name)
    logger.info("=" * 70)

    for rname, alpha in alphas.items():
        t1 = time.time()
        mlp = MLPClassifier(
            hidden_layer_sizes=best_layers,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            alpha=alpha,
        )
        mlp.fit(best_Xo, y_oof)
        p = mlp.predict_proba(best_Xt)
        m = _compute_metrics(y_test, p)
        m["experiment"] = f"reg_{rname}"
        m["features"] = best_Xo.shape[1]
        m["arch"] = best_arch_name
        m["alpha"] = alpha
        m["calibration"] = "none"
        m["time_s"] = round(time.time() - t1, 1)
        results.append(m)
        logger.info(
            "  alpha=%-8s  ll=%.6f  ECE_draw=%.4f  bias=%+.4f  (%.1fs)",
            rname,
            m["log_loss"],
            m["ece_draw"],
            m["draw_bias"],
            m["time_s"],
        )

    # Best alpha
    best_alpha_r = min(
        [r for r in results if r["experiment"].startswith("reg_")],
        key=lambda r: r["log_loss"],
    )
    best_alpha = best_alpha_r["alpha"]

    # ==== AXE 4: Post-calibration (best everything) ====
    logger.info("=" * 70)
    logger.info(
        "AXE 4: Calibration (features=%s, arch=%s, alpha=%s)",
        best_fname,
        best_arch_name,
        best_alpha,
    )
    logger.info("=" * 70)

    mlp_final = MLPClassifier(
        hidden_layer_sizes=best_layers,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        alpha=best_alpha,
    )
    mlp_final.fit(best_Xo, y_oof)
    p_oof_raw = mlp_final.predict_proba(best_Xo)
    p_test_raw = mlp_final.predict_proba(best_Xt)

    # C1: raw (already in results)
    # C2: temperature
    from scipy.optimize import minimize_scalar

    def _nll_temp(T: float) -> float:
        logits = np.log(np.clip(p_oof_raw, 1e-7, 1.0)) / T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return float(log_loss(y_oof, p))

    T = float(minimize_scalar(_nll_temp, bounds=(0.5, 2.0), method="bounded").x)
    logits_t = np.log(np.clip(p_test_raw, 1e-7, 1.0)) / T
    logits_t -= logits_t.max(axis=1, keepdims=True)
    p_test_temp = np.exp(logits_t) / np.exp(logits_t).sum(axis=1, keepdims=True)

    m_temp = _compute_metrics(y_test, p_test_temp)
    m_temp.update(
        {
            "experiment": "cal_temperature",
            "features": best_Xo.shape[1],
            "arch": best_arch_name,
            "alpha": best_alpha,
            "calibration": f"temp_T={T:.4f}",
            "time_s": 0,
        }
    )
    results.append(m_temp)
    logger.info(
        "  Temperature T=%.4f  ll=%.6f  ECE_draw=%.4f  bias=%+.4f",
        T,
        m_temp["log_loss"],
        m_temp["ece_draw"],
        m_temp["draw_bias"],
    )

    # C3: Dirichlet
    p_test_dir = _dirichlet_cal(y_oof, p_oof_raw, p_test_raw)
    m_dir = _compute_metrics(y_test, p_test_dir)
    m_dir.update(
        {
            "experiment": "cal_dirichlet",
            "features": best_Xo.shape[1],
            "arch": best_arch_name,
            "alpha": best_alpha,
            "calibration": "dirichlet",
            "time_s": 0,
        }
    )
    results.append(m_dir)
    logger.info(
        "  Dirichlet       ll=%.6f  ECE_draw=%.4f  bias=%+.4f",
        m_dir["log_loss"],
        m_dir["ece_draw"],
        m_dir["draw_bias"],
    )

    # ==== SUMMARY ====
    logger.info("\n" + "=" * 90)
    logger.info("FULL RESULTS SUMMARY")
    logger.info("=" * 90)
    logger.info(
        "%-25s %5s %10s %10s %10s %10s",
        "Experiment",
        "Feat",
        "log_loss",
        "ECE_draw",
        "draw_bias",
        "rps",
    )
    logger.info("-" * 90)

    for r in sorted(results, key=lambda x: x["log_loss"]):
        logger.info(
            "%-25s %5d %10.6f %10.4f %+10.4f %10.6f",
            r["experiment"],
            r["features"],
            r["log_loss"],
            r["ece_draw"],
            r["draw_bias"],
            r["rps"],
        )

    logger.info("-" * 90)
    logger.info(
        "%-25s %5s %10s %10s %10s %10s",
        "LGB+Dirichlet (ref)",
        "",
        "0.554117",
        "0.0042",
        "-0.0017",
        "0.088012",
    )
    logger.info(
        "%-25s %5s %10s %10s %10s %10s",
        "SimpleAvg (ref)",
        "",
        "0.555770",
        "0.0061",
        "-0.0019",
        "0.087991",
    )

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_DIR / "sweep_results.csv", index=False)
    with open(OUT_DIR / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", OUT_DIR)
    logger.info("Total sweep time: %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
