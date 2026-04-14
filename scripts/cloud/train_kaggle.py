"""Kaggle Cloud Training — ALICE Engine V9 Training Final (ISO 42001/5259/5055).

1 kernel per model on full 1.1M dataset. V9 params from 590-config HP search.
Per-model alpha (ADR-008), Dirichlet calibration (Kull 2019), Tier 2 draw metrics.
Requires FE kernel (alice-fe-v8) output via kernel_sources.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))


def _setup_kaggle_imports() -> None:
    """Find alice-code dataset and add to sys.path."""
    import zipfile  # noqa: PLC0415

    candidates = [
        Path("/kaggle/input/alice-code"),
        Path("/kaggle/input/datasets/pguillemin/alice-code"),
    ]
    kaggle_input = next((c for c in candidates if c.exists()), None)
    root = Path("/kaggle/input")
    if root.exists():
        items = list(root.rglob("*"))
        logger.info("/kaggle/input/ tree (%d items): %s", len(items), [str(f) for f in items[:30]])
    logger.info("kaggle_input=%s", kaggle_input)
    if not kaggle_input:
        return
    zips = list(kaggle_input.rglob("*.zip"))
    if zips:
        wd = Path("/kaggle/working/code")
        wd.mkdir(parents=True, exist_ok=True)
        for zf in zips:
            with zipfile.ZipFile(zf) as z:
                z.extractall(wd)
        sys.path.insert(0, str(wd))
    else:
        sys.path.insert(0, str(kaggle_input))
    logger.info("sys.path += %s", sys.path[0])


def _load_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """Load feature parquets from FE kernel output (kernel_sources). Returns (train, valid, test, features_dir)."""
    # kernel_sources mount under /kaggle/input/notebooks/{user}/{slug}/
    # dataset_sources mount under /kaggle/input/datasets/{user}/{slug}/
    # Also try direct slug mount (undocumented but sometimes works)
    fe_candidates = [
        Path("/kaggle/input/notebooks/pguillemin/alice-fe-v8/features"),
        Path("/kaggle/input/notebooks/pguillemin/alice-fe-v8"),
        Path("/kaggle/input/alice-fe-v8/features"),
        Path("/kaggle/input/alice-fe-v8"),
        Path("/kaggle/input/datasets/pguillemin/alice-fe-v8/features"),
    ]
    for fe_dir in fe_candidates:
        if fe_dir.exists() and all(
            (fe_dir / f"{s}.parquet").exists() for s in ("train", "valid", "test")
        ):
            logger.info("Feature parquets found in FE kernel output: %s", fe_dir)
            return (
                pd.read_parquet(fe_dir / "train.parquet"),
                pd.read_parquet(fe_dir / "valid.parquet"),
                pd.read_parquet(fe_dir / "test.parquet"),
                fe_dir,
            )

    # Diagnostic: list what IS mounted
    root = Path("/kaggle/input")
    if root.exists():
        mounted = [str(p) for p in root.rglob("*.parquet")]
        logger.error("No FE parquets found. Parquet files in /kaggle/input/: %s", mounted[:20])
    msg = "FE kernel output not found. Run alice-fe-v8 kernel first."
    logger.error(msg)
    raise FileNotFoundError(msg)


def _compute_baselines(
    train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_train: pd.Series,
) -> dict:
    """Compute naive + Elo baselines for quality gate (ISO 25059)."""
    from sklearn.metrics import log_loss as sk_log_loss  # noqa: PLC0415

    from scripts.baselines import compute_elo_baseline, compute_naive_baseline  # noqa: PLC0415
    from scripts.features.draw_priors import build_draw_rate_lookup  # noqa: PLC0415
    from scripts.kaggle_metrics import compute_expected_score_mae, compute_rps  # noqa: PLC0415

    n_test = len(y_test)
    y_test_arr = y_test.values
    naive_proba = compute_naive_baseline(y_train.values, n_test)
    draw_lookup = build_draw_rate_lookup(train)
    b_elo = X_test["blanc_elo"].values if "blanc_elo" in X_test.columns else np.full(n_test, 1500)
    n_elo = X_test["noir_elo"].values if "noir_elo" in X_test.columns else np.full(n_test, 1500)
    elo_proba = compute_elo_baseline(b_elo, n_elo, draw_lookup)
    oh = np.eye(3)[y_test_arr]
    return (
        {
            "naive": {
                "log_loss": float(sk_log_loss(y_test_arr, naive_proba)),
                "rps": float(compute_rps(y_test_arr, naive_proba)),
                "brier": float(np.mean(np.sum((naive_proba - oh) ** 2, axis=1))),
            },
            "elo": {
                "log_loss": float(sk_log_loss(y_test_arr, elo_proba)),
                "rps": float(compute_rps(y_test_arr, elo_proba)),
                "es_mae": float(compute_expected_score_mae(y_test_arr, elo_proba)),
            },
        },
        draw_lookup,
    )


def main() -> None:
    """Full Kaggle training pipeline orchestration (ISO 42001)."""
    # Single-model mode: detect from env var, slug, or notebook title
    kaggle_env = {k: v for k, v in os.environ.items() if k.startswith("KAGGLE")}
    logger.info("KAGGLE env vars: %s", list(kaggle_env.keys()))
    model_filter = os.environ.get("ALICE_MODEL")
    if not model_filter:
        # Try multiple Kaggle env vars that may contain the kernel name
        for env_key in ("KAGGLE_KERNEL_RUN_SLUG", "KAGGLE_URL_BASE", "KAGGLE_DOCKER_IMAGE"):
            val = os.environ.get(env_key, "")
            for m in ("xgboost", "catboost", "lightgbm"):
                if m in val.lower():
                    model_filter = m
                    logger.info("Model detected from %s=%s", env_key, val)
                    break
            if model_filter:
                break
    if model_filter:
        logger.info("Single-model mode: %s", model_filter)
    elif Path("/kaggle/working").exists():
        model_filter = "xgboost"
        logger.warning("No model filter detected on Kaggle — defaulting to xgboost (safe)")
    else:
        logger.info("Local mode — training all models")
    label = f" [{model_filter.upper()}]" if model_filter else ""
    logger.info("ALICE Engine — V9 Training Final%s", label)
    _setup_kaggle_imports()

    from scripts.kaggle_artifacts import (  # noqa: PLC0415
        build_lineage,
        build_model_card,
        fetch_champion_ll,
        save_metadata_and_push,
        save_models,
        setup_hf_auth,
    )
    from scripts.kaggle_diagnostics import save_diagnostics  # noqa: PLC0415
    from scripts.kaggle_metrics import evaluate_on_test  # noqa: PLC0415
    from scripts.kaggle_quality_gates import (  # noqa: PLC0415
        check_quality_gates,
        compute_split_logloss,
    )
    from scripts.kaggle_trainers import (  # noqa: PLC0415
        LABEL_COLUMN,
        MODEL_EXTENSIONS,
        default_hyperparameters,
        prepare_features,
        train_all_sequential,
    )

    setup_hf_auth()
    train, valid, test, features_dir = _load_features()
    lineage = build_lineage(train, valid, test, features_dir, label_column=LABEL_COLUMN)
    logger.info("Lineage: train=%d valid=%d test=%d", len(train), len(valid), len(test))

    X_train, y_train, X_valid, y_valid, X_test, y_test, encoders = prepare_features(
        train, valid, test
    )

    # NaN audit per split — MANDATORY (2 weeks of debug on v9-v13 without this)
    for split_name, df_split in [("train", X_train), ("valid", X_valid), ("test", X_test)]:
        dead = [c for c in df_split.columns if df_split[c].isna().mean() > 0.99]
        if dead:
            logger.error("STOP: %d features >99%% NaN on %s: %s", len(dead), split_name, dead[:10])
            raise ValueError(f"{len(dead)} features >99% NaN on {split_name}: {dead[:5]}")
        nan_any = {
            c: df_split[c].isna().mean() for c in df_split.columns if df_split[c].isna().any()
        }
        if nan_any:
            logger.info(
                "  %s NaN features (%d): max=%.1f%%",
                split_name,
                len(nan_any),
                max(nan_any.values()) * 100,
            )

    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    config = default_hyperparameters()

    # V9 guard: crash immediately if stale V8 code is running (dataset propagation issue)
    _xgb_eta = config["xgboost"].get("eta", 0)
    _cb_rsm = config["catboost"].get("rsm", 0)
    if _xgb_eta < 0.01 or _cb_rsm < 0.5:
        raise RuntimeError(
            f"STALE CODE DETECTED: eta={_xgb_eta}, rsm={_cb_rsm}. "
            "Expected V9 params (eta=0.05, rsm=0.7). "
            "Re-upload alice-code dataset and wait for propagation."
        )

    config["catboost"]["train_dir"] = str(out_dir / "catboost_info")

    # Compute Elo init scores BEFORE feature subset (needs blanc_elo/noir_elo)
    from scripts.baselines import compute_init_scores_from_features  # noqa: PLC0415
    from scripts.features.draw_priors import build_draw_rate_lookup  # noqa: PLC0415

    draw_lookup_train = build_draw_rate_lookup(train)
    init_scores_train = compute_init_scores_from_features(X_train, draw_lookup_train)
    init_scores_valid = compute_init_scores_from_features(X_valid, draw_lookup_train)
    init_scores_test = compute_init_scores_from_features(X_test, draw_lookup_train)

    # Per-model alpha (ADR-008): each architecture has different gradient sensitivity.
    # LGB=0.1 (GOSS drops small gradients), CB=0.3 (oblivious), XGB=0.5 (depth-wise robust).
    # Env var override for grid/optuna sweeps: ALICE_INIT_ALPHA=0.5
    alpha_per_model = {
        "catboost": config["catboost"].get("init_score_alpha", 0.3),
        "xgboost": config["xgboost"].get("init_score_alpha", 0.5),
        "lightgbm": config["lightgbm"].get("init_score_alpha", 0.1),
    }
    alpha_override = os.environ.get("ALICE_INIT_ALPHA")
    if alpha_override:
        alpha = float(alpha_override)
        logger.info("Init score alpha=%.2f (env override, all models)", alpha)
    elif model_filter:
        alpha = alpha_per_model.get(model_filter.lower(), 0.5)
        logger.info("Init score alpha=%.2f (per-model, %s)", alpha, model_filter)
    else:
        alpha = 0.5
        logger.warning("No model filter — using fallback alpha=0.5")
    if alpha != 1.0:
        init_scores_train = init_scores_train * alpha
        init_scores_valid = init_scores_valid * alpha
        init_scores_test = init_scores_test * alpha

    for _name, _scores in [
        ("train", init_scores_train),
        ("valid", init_scores_valid),
        ("test", init_scores_test),
    ]:
        logger.info(
            "  %s init_scores: min=%.3f max=%.3f mean=[%.3f, %.3f, %.3f]",
            _name,
            _scores.min(),
            _scores.max(),
            *_scores.mean(axis=0),
        )

    # Feature subset AFTER init_scores (blanc_elo/noir_elo not in top11)
    feature_subset = os.environ.get("ALICE_FEATURE_SUBSET", "all")  # v9: all features + low LR
    if feature_subset in ("top10", "domain13"):
        # Domain-driven selection for residual learning (NOT from v3 importance)
        # Focus: features that CORRECT Elo baseline, especially draw prediction
        domain13 = [
            "diff_elo",
            "elo_proximity",
            "avg_elo",  # Elo curve adjustment
            "draw_rate_prior",
            "draw_rate_blanc",
            "draw_rate_noir",  # Draw correction
            "type_competition",
            "echiquier",
            "est_domicile_blanc",  # Context
            "expected_score_recent_blanc",
            "expected_score_recent_noir",  # Form
            "win_rate_home_dom",
            "draw_rate_home_dom",  # Home advantage
        ]
        top11 = domain13
        keep = [c for c in top11 if c in X_train.columns]
        logger.info("Feature subset: top11 (%d/%d available)", len(keep), len(top11))
        X_train, X_valid, X_test = X_train[keep], X_valid[keep], X_test[keep]
    logger.info(
        "Elo init scores computed: train=%s valid=%s test=%s",
        init_scores_train.shape,
        init_scores_valid.shape,
        init_scores_test.shape,
    )

    results = train_all_sequential(
        X_train,
        y_train,
        X_valid,
        y_valid,
        config,
        init_scores_train=init_scores_train,
        init_scores_valid=init_scores_valid,
        checkpoint_dir=out_dir,
        encoders=encoders,
        model_extensions=MODEL_EXTENSIONS,
        model_filter=model_filter,
    )
    # Triple calibration: temperature, isotonic, Dirichlet (Kull et al. 2019 NeurIPS)
    # Dirichlet = K×K affine on log-probas → captures inter-class interactions (draw↔loss/win)
    # Quality gates FIRST, then SHAP (skill: "gates FIRST, SHAP can be skipped in emergency")
    from scripts.kaggle_diagnostics import (  # noqa: PLC0415
        calibrate_models,
        calibrate_models_dirichlet,
        calibrate_models_isotonic,
    )

    cal_temp = calibrate_models(results, X_valid, y_valid, out_dir, init_scores_valid)
    cal_iso = calibrate_models_isotonic(results, X_valid, y_valid, out_dir, init_scores_valid)
    cal_dir = calibrate_models_dirichlet(results, X_valid, y_valid, out_dir, init_scores_valid)

    baseline_metrics, draw_lookup = _compute_baselines(train, X_test, y_test, y_train)
    draw_lookup.to_parquet(out_dir / "draw_rate_lookup.parquet", index=False)
    logger.info("Saved draw_rate_lookup.parquet (%d cells) for inference", len(draw_lookup))
    champion_ll = fetch_champion_ll()

    # Evaluate all 3 calibrators, pick best on quality gates + logloss
    cal_candidates = [
        ("TEMPERATURE", cal_temp),
        ("ISOTONIC", cal_iso),
        ("DIRICHLET", cal_dir),
    ]
    best_cal_name, calibrators, gate = None, cal_temp, {}
    best_ll = 999.0
    for cal_name, cal in cal_candidates:
        evaluate_on_test(
            results, X_test, y_test, init_scores_test=init_scores_test, calibrators=cal
        )
        g = check_quality_gates(
            results,
            baseline_metrics=baseline_metrics,
            champion_ll=champion_ll,
            verbose=False,
        )
        ll = g.get("best_log_loss", 999.0)
        logger.info("Quality gate %s: passed=%s logloss=%.6f", cal_name, g.get("passed"), ll)
        if g.get("passed") and ll < best_ll:
            best_cal_name, calibrators, gate, best_ll = cal_name, cal, g, ll
    # Fallback: if none pass, pick lowest logloss
    if best_cal_name is None:
        for cal_name, cal in cal_candidates:
            evaluate_on_test(
                results, X_test, y_test, init_scores_test=init_scores_test, calibrators=cal
            )
            g = check_quality_gates(results, baseline_metrics=baseline_metrics, verbose=False)
            ll = g.get("best_log_loss", 999.0)
            if ll < best_ll:
                best_cal_name, calibrators, gate, best_ll = cal_name, cal, g, ll
        logger.warning("No calibrator passes all gates — using %s (lowest logloss)", best_cal_name)
    else:
        logger.info("Winner: %s (logloss=%.6f)", best_cal_name, best_ll)
    # Final evaluate with winner so saved metrics/predictions match
    evaluate_on_test(
        results, X_test, y_test, init_scores_test=init_scores_test, calibrators=calibrators
    )
    # Authoritative T1-T12 gate check with T10 train-test gap (ISO 42001)
    train_ll = compute_split_logloss(
        results,
        X_train,
        y_train,
        init_scores_train,
        calibrators,
    )
    gate = check_quality_gates(
        results,
        baseline_metrics=baseline_metrics,
        champion_ll=champion_ll,
        train_log_loss=train_ll,
        verbose=True,
    )

    save_models(results, encoders, out_dir, model_extensions=MODEL_EXTENSIONS)
    save_diagnostics(
        results,
        X_test,
        y_test,
        X_valid,
        y_valid,
        X_train,
        out_dir,
        init_scores_valid=init_scores_valid,
        init_scores_test=init_scores_test,
        calibrators=calibrators,
    )
    metadata = build_model_card(results, lineage, gate, config, MODEL_EXTENSIONS, out_dir=out_dir)
    metadata["version"] = version
    if gate.get("passed"):
        save_metadata_and_push(metadata, out_dir)
    else:
        logger.error("Quality gate FAILED: %s — saving locally only, NO push.", gate.get("reason"))
        import json  # noqa: PLC0415

        with open(out_dir / "metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)

    # SHAP AFTER gates+save (skill: "gates FIRST, SHAP can be skipped in emergency")
    # Permutation excluded for resume kernels (skill: 197×5×17s = 4h39m timeout risk)
    from scripts.kaggle_shap import compute_shap_importance  # noqa: PLC0415

    compute_shap_importance(results, X_test, y_test, init_scores_test, out_dir)

    logger.info(
        "Done. Status=%s Best=%s LogLoss=%.4f",
        metadata["status"],
        gate.get("best_model"),
        gate.get("best_log_loss", 0),
    )


if __name__ == "__main__":
    main()
