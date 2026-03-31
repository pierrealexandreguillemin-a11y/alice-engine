"""Resume XGBoost from checkpoint — ISO eval pipeline (CPU, 8-11h).

A: preflight (SHA-256, NaN, sanity predict). B: resume (eta=0.01, early_stop=200,
save_best=True, TrainingCheckPoint/5000). C: eval (TreeSHAP, calibration, gates,
diagnostics, model card). Permutation importance excluded (4-5h) — TreeSHAP instead.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))


def _setup_kaggle_imports() -> None:
    """Find alice-code dataset and add to sys.path (handles ZIP)."""
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
    """Load FE kernel output parquets. Returns (train, valid, test, features_dir)."""
    fe_candidates = [
        Path("/kaggle/input/notebooks/pguillemin/alice-fe-v8/features"),
        Path("/kaggle/input/notebooks/pguillemin/alice-fe-v8"),
        Path("/kaggle/input/alice-fe-v8/features"),
        Path("/kaggle/input/alice-fe-v8"),
    ]
    for fe_dir in fe_candidates:
        if fe_dir.exists() and all(
            (fe_dir / f"{s}.parquet").exists() for s in ("train", "valid", "test")
        ):
            logger.info("Features: %s", fe_dir)
            return (
                pd.read_parquet(fe_dir / "train.parquet"),
                pd.read_parquet(fe_dir / "valid.parquet"),
                pd.read_parquet(fe_dir / "test.parquet"),
                fe_dir,
            )
    msg = "FE kernel output not found"
    raise FileNotFoundError(msg)


def _find_checkpoint() -> tuple[Path, str]:
    """Find checkpoint .ubj and compute SHA-256. Returns (path, sha256_hex)."""
    roots = [
        Path("/kaggle/input/datasets/pguillemin/alice-xgboost-checkpoint"),
        Path("/kaggle/input/alice-xgboost-checkpoint"),
        Path("/kaggle/input/notebooks/pguillemin/alice-train-xgboost"),
    ]
    for root in roots:
        for ckpt in root.rglob("xgboost_checkpoint.ubj") if root.exists() else []:
            sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
            logger.info(
                "Checkpoint: %s (%.1f MB, SHA256=%s)", ckpt, ckpt.stat().st_size / 1e6, sha[:16]
            )
            return ckpt, sha
    msg = "xgboost_checkpoint.ubj not found"
    raise FileNotFoundError(msg)


def _preflight(booster: object, X_train: pd.DataFrame) -> None:
    """Validate checkpoint integrity + data coherence before training."""
    import xgboost as xgb  # noqa: PLC0415

    expected = int(os.environ.get("ALICE_CKPT_ROUNDS", "50000"))
    actual = booster.num_boosted_rounds()
    if actual != expected:
        logger.warning("Checkpoint rounds: %d (expected %d)", actual, expected)
    # Sanity predict on 10 rows
    dm = xgb.DMatrix(X_train.iloc[:10])
    preds = np.asarray(booster.predict(dm)).reshape(-1, 3)
    assert not np.any(np.isnan(preds)), "NaN in sanity predictions"
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-5), "Probas don't sum to 1"
    assert np.all(preds >= 0) and np.all(preds <= 1), "Probas out of [0,1]"
    logger.info("Preflight OK: %d rounds, sanity predict clean", actual)


def _compute_baselines(
    train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, y_train: pd.Series
) -> tuple:
    """Compute naive + Elo baselines for quality gate (ISO 25059)."""
    from sklearn.metrics import log_loss as sk_log_loss  # noqa: PLC0415

    from scripts.baselines import compute_elo_baseline, compute_naive_baseline  # noqa: PLC0415
    from scripts.features.draw_priors import build_draw_rate_lookup  # noqa: PLC0415
    from scripts.kaggle_metrics import compute_expected_score_mae, compute_rps  # noqa: PLC0415

    y_arr, n = y_test.values, len(y_test)
    naive = compute_naive_baseline(y_train.values, n)
    draw_lookup = build_draw_rate_lookup(train)
    b_elo = X_test["blanc_elo"].values if "blanc_elo" in X_test.columns else np.full(n, 1500)
    n_elo = X_test["noir_elo"].values if "noir_elo" in X_test.columns else np.full(n, 1500)
    elo = compute_elo_baseline(b_elo, n_elo, draw_lookup)
    oh = np.eye(3)[y_arr]
    return (
        {
            "naive": {
                "log_loss": float(sk_log_loss(y_arr, naive)),
                "rps": float(compute_rps(y_arr, naive)),
                "brier": float(np.mean(np.sum((naive - oh) ** 2, axis=1))),
            },
            "elo": {
                "log_loss": float(sk_log_loss(y_arr, elo)),
                "rps": float(compute_rps(y_arr, elo)),
                "es_mae": float(compute_expected_score_mae(y_arr, elo)),
            },
        },
        draw_lookup,
    )


def main() -> None:  # noqa: PLR0915
    """Resume XGBoost from checkpoint with full ISO pipeline (8-11h CPU)."""
    t_start = time.time()
    _setup_kaggle_imports()
    import xgboost as xgb

    from scripts.baselines import compute_init_scores_from_features
    from scripts.features.draw_priors import build_draw_rate_lookup
    from scripts.kaggle_artifacts import (
        build_lineage,
        build_model_card,
        fetch_champion_ll,
        save_metadata_and_push,
        save_models,
        setup_hf_auth,
    )
    from scripts.kaggle_diagnostics import (
        calibrate_models,
        calibrate_models_isotonic,
        save_diagnostics,
    )
    from scripts.kaggle_metrics import XGBWrapper, check_quality_gates, evaluate_on_test
    from scripts.kaggle_trainers import (
        LABEL_COLUMN,
        MODEL_EXTENSIONS,
        default_hyperparameters,
        prepare_features,
    )

    setup_hf_auth()

    # === PHASE A: PREFLIGHT ===
    train, valid, test, features_dir = _load_features()
    lineage = build_lineage(train, valid, test, features_dir, label_column=LABEL_COLUMN)
    logger.info("Lineage: train=%d valid=%d test=%d", len(train), len(valid), len(test))
    X_train, y_train, X_valid, y_valid, X_test, y_test, encoders = prepare_features(
        train, valid, test
    )

    # NaN preflight (ISO 5259 — 61 features mortes si split exclut saison courante)
    for split_name, df_split in [("train", X_train), ("valid", X_valid), ("test", X_test)]:
        dead = [c for c in df_split.columns if df_split[c].isna().mean() > 0.99]
        if dead:
            logger.error(
                "CRITICAL: %d features >99%% NaN on %s: %s", len(dead), split_name, dead[:5]
            )
            msg = f"{len(dead)} features >99% NaN on {split_name} — split bug, not dead features"
            raise ValueError(msg)

    config = default_hyperparameters()
    alpha = float(os.environ.get("ALICE_INIT_ALPHA", config["global"]["init_score_alpha"]))
    draw_lookup = build_draw_rate_lookup(train)
    init_train = compute_init_scores_from_features(X_train, draw_lookup) * alpha
    init_valid = compute_init_scores_from_features(X_valid, draw_lookup) * alpha
    init_test = compute_init_scores_from_features(X_test, draw_lookup) * alpha
    logger.info("Init scores alpha=%.2f, test mean=%s", alpha, init_test.mean(axis=0).round(3))

    ckpt_path, ckpt_sha = _find_checkpoint()
    booster = xgb.Booster()
    booster.load_model(str(ckpt_path))
    _preflight(booster, X_train)

    # === PHASE B: RESUME TRAINING (eta=0.01) ===
    extra_rounds = int(os.environ.get("ALICE_EXTRA_ROUNDS", "50000"))
    early_stop = int(os.environ.get("ALICE_EARLY_STOP", "200"))
    resume_eta = float(os.environ.get("ALICE_RESUME_ETA", "0.01"))

    xgb_p = {
        k: v
        for k, v in config["xgboost"].items()
        if k not in ("n_estimators", "early_stopping_rounds")
    }
    xgb_p["eta"] = resume_eta  # 2x original: faster convergence, save_best handles early stop
    n_before = booster.num_boosted_rounds()
    logger.info(
        "Resuming: +%d rounds (eta=%.3f, early_stop=%d) from %d",
        extra_rounds,
        resume_eta,
        early_stop,
        n_before,
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtrain.set_base_margin(init_train.ravel())
    dvalid.set_base_margin(init_valid.ravel())

    t0 = time.time()
    evals_log: dict = {}
    ckpt_interval = int(os.environ.get("ALICE_CKPT_INTERVAL", "5000"))
    ckpt_dir = OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    es_cb = xgb.callback.EarlyStopping(rounds=early_stop, save_best=True)
    ckpt_cb = xgb.callback.TrainingCheckPoint(
        directory=str(ckpt_dir), name="xgb_resume", interval=ckpt_interval
    )
    logger.info("Checkpoints every %d rounds to %s", ckpt_interval, ckpt_dir)
    booster = xgb.train(
        xgb_p,
        dtrain,
        num_boost_round=extra_rounds,
        evals=[(dvalid, "val")],
        verbose_eval=500,
        xgb_model=booster,
        evals_result=evals_log,
        callbacks=[es_cb, ckpt_cb],
    )
    train_time = time.time() - t0
    n_after = booster.num_boosted_rounds()
    best_iter = getattr(booster, "best_iteration", n_after - 1)
    best_score = getattr(booster, "best_score", -1.0)
    logger.info(
        "Done: %d->%d rounds (+%d) in %.0fs. best_iter=%d, best_score=%.6f",
        n_before,
        n_after,
        n_after - n_before,
        train_time,
        best_iter,
        best_score,
    )
    del dtrain, dvalid
    gc.collect()

    # === PHASE C: ESSENTIAL EVAL (calibration + gates + model card) ===
    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "xgboost_checkpoint.ubj"))
    logger.info("Checkpoint saved: %s", out_dir / "xgboost_checkpoint.ubj")

    wrapper = XGBWrapper(booster, X_train.columns, 3, evals_result=evals_log)
    feat_gain = booster.get_score(importance_type="gain")
    results = {
        "XGBoost": {
            "model": wrapper,
            "metrics": {"train_time_s": train_time},
            "importance": feat_gain,
        }
    }

    # XGBoost gain + TreeSHAP importance (both instant — permutation excluded)
    gain_df = pd.DataFrame(
        sorted(feat_gain.items(), key=lambda x: -x[1]),
        columns=["feature", "gain"],
    )
    gain_df.to_csv(out_dir / "xgboost_gain_importance.csv", index=False)
    n_active = int((gain_df["gain"] > 0).sum())
    logger.info("Gain importance: %d/%d features active", n_active, len(gain_df))

    # TreeSHAP (Lundberg & Lee 2017) — XGBoost native, O(TLD) not O(TL2^M)
    dtest_shap = xgb.DMatrix(X_test)
    dtest_shap.set_base_margin(init_test.ravel())
    shap_vals = booster.predict(dtest_shap, pred_contribs=True)  # (N, F+1, 3)
    shap_vals = shap_vals.reshape(len(X_test), -1, 3)[:, :-1, :]  # drop bias
    mean_shap = np.abs(shap_vals).mean(axis=0).sum(axis=1)  # mean |SHAP| per feature
    shap_df = pd.DataFrame({"feature": list(X_test.columns), "treeshap": mean_shap})
    shap_df = shap_df.sort_values("treeshap", ascending=False).reset_index(drop=True)
    shap_df.to_csv(out_dir / "xgboost_treeshap_importance.csv", index=False)
    logger.info("TreeSHAP top 5: %s", list(shap_df.head(5).itertuples(index=False, name=None)))
    del dtest_shap, shap_vals
    gc.collect()

    cal_temp = calibrate_models(results, X_valid, y_valid, out_dir, init_valid)
    cal_iso = calibrate_models_isotonic(results, X_valid, y_valid, out_dir, init_valid)

    baseline_metrics, bl_draw_lookup = _compute_baselines(train, X_test, y_test, y_train)
    bl_draw_lookup.to_parquet(out_dir / "draw_rate_lookup.parquet", index=False)
    champion_ll = fetch_champion_ll()

    evaluate_on_test(results, X_test, y_test, init_scores_test=init_test, calibrators=cal_temp)
    gate_temp = check_quality_gates(
        results, baseline_metrics=baseline_metrics, champion_ll=champion_ll
    )
    logger.info("TEMPERATURE gate: %s", gate_temp)
    evaluate_on_test(results, X_test, y_test, init_scores_test=init_test, calibrators=cal_iso)
    gate_iso = check_quality_gates(
        results, baseline_metrics=baseline_metrics, champion_ll=champion_ll
    )
    logger.info("ISOTONIC gate: %s", gate_iso)

    # Pick winner (same logic as train_kaggle.py)
    if gate_iso.get("passed") and not gate_temp.get("passed"):
        calibrators, gate = cal_iso, gate_iso
    elif gate_temp.get("passed") and gate_iso.get("passed"):
        ll_t, ll_i = gate_temp.get("best_log_loss", 1.0), gate_iso.get("best_log_loss", 1.0)
        calibrators, gate = (cal_temp, gate_temp) if ll_t <= ll_i else (cal_iso, gate_iso)
    else:
        calibrators, gate = cal_temp, gate_temp
    evaluate_on_test(results, X_test, y_test, init_scores_test=init_test, calibrators=calibrators)

    save_models(results, encoders, out_dir, model_extensions=MODEL_EXTENSIONS)

    metadata = build_model_card(results, lineage, gate, config, MODEL_EXTENSIONS, out_dir=out_dir)
    metadata["version"] = version
    metadata["status"] = "RESUMED"
    metadata["resume_info"] = {
        "checkpoint_sha256": ckpt_sha,
        "rounds_before": n_before,
        "rounds_after": n_after,
        "rounds_added": n_after - n_before,
        "resume_eta": resume_eta,
        "best_iteration": best_iter,
        "best_score": best_score,
        "train_time_s": round(train_time, 1),
    }
    # Git commit hash (ISO 5259 traceability)
    import subprocess  # noqa: PLC0415

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S603, S607
    except Exception:
        git_hash = os.environ.get("KAGGLE_KERNEL_RUN_SLUG", "unknown")
    metadata["git_commit"] = git_hash
    if gate.get("passed"):
        save_metadata_and_push(metadata, out_dir)
    else:
        logger.error("Quality gate FAILED: %s — saving locally only.", gate.get("reason"))
        with open(out_dir / "metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)

    m = results["XGBoost"]["metrics"]
    logger.info(
        "RESULT: gate=%s ll=%.5f rps=%.5f es_mae=%.5f (%d->%d rounds, eta=%.3f)",
        "PASS" if gate.get("passed") else "FAIL",
        m.get("test_log_loss", 0),
        m.get("test_rps", 0),
        m.get("test_es_mae", 0),
        n_before,
        n_after,
        resume_eta,
    )

    # Diagnostics ISO (ROC, calibration curves, confusion matrices, distributions)
    save_diagnostics(
        results,
        X_test,
        y_test,
        X_valid,
        y_valid,
        X_train,
        out_dir,
        init_scores_valid=init_valid,
        init_scores_test=init_test,
        calibrators=calibrators,
    )
    logger.info("Total elapsed: %.1fh", (time.time() - t_start) / 3600)


if __name__ == "__main__":
    main()
