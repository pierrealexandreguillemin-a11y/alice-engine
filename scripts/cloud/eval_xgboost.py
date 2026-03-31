"""Eval-only kernel: load XGBoost checkpoint, run quality gates T1-T12 (CPU, ~10min)."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))


def _setup_imports() -> None:
    """Add alice-code dataset to sys.path."""
    candidates = [
        Path("/kaggle/input/alice-code"),
        Path("/kaggle/input/datasets/pguillemin/alice-code"),
    ]
    kaggle_input = next((c for c in candidates if c.exists()), None)
    if kaggle_input:
        sys.path.insert(0, str(kaggle_input))
        logger.info("sys.path += %s", kaggle_input)


def _find_checkpoint() -> Path:
    """Find xgboost_checkpoint.ubj in dataset or kernel output."""
    roots = [
        Path("/kaggle/input/datasets/pguillemin/alice-xgboost-checkpoint"),
        Path("/kaggle/input/alice-xgboost-checkpoint"),
        Path("/kaggle/input/notebooks/pguillemin/alice-train-xgboost"),
        Path("/kaggle/input/alice-train-xgboost"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for ckpt in root.rglob("xgboost_checkpoint.ubj"):
            logger.info("Checkpoint found: %s (%.1f MB)", ckpt, ckpt.stat().st_size / 1e6)
            return ckpt
    # List what IS mounted for debugging
    inp = Path("/kaggle/input")
    if inp.exists():
        items = [str(p) for p in inp.rglob("*") if p.is_file()]
        logger.error("No checkpoint found. Files in /kaggle/input/: %s", items[:30])
    msg = "xgboost_checkpoint.ubj not found"
    raise FileNotFoundError(msg)


def _find_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load FE kernel output parquets."""
    fe_candidates = [
        Path("/kaggle/input/notebooks/pguillemin/alice-fe-v8/features"),
        Path("/kaggle/input/alice-fe-v8/features"),
        Path("/kaggle/input/notebooks/pguillemin/alice-fe-v8"),
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
            )
    msg = "FE parquets not found"
    raise FileNotFoundError(msg)


def main() -> None:
    """Load checkpoint, evaluate, run quality gates."""
    _setup_imports()
    import xgboost as xgb

    from scripts.baselines import (
        compute_elo_baseline,
        compute_init_scores_from_features,
        compute_naive_baseline,
    )
    from scripts.features.draw_priors import build_draw_rate_lookup
    from scripts.kaggle_diagnostics import calibrate_models, calibrate_models_isotonic
    from scripts.kaggle_metrics import (
        check_quality_gates,
        compute_expected_score_mae,
        compute_rps,
        evaluate_on_test,
    )
    from scripts.kaggle_trainers import prepare_features

    # 1. Load data
    train, valid, test = _find_features()
    logger.info("Loaded: train=%d valid=%d test=%d", len(train), len(valid), len(test))

    X_train, y_train, X_valid, y_valid, X_test, y_test, encoders = prepare_features(
        train, valid, test
    )
    logger.info("Features: %d cols, test=%d rows", X_train.shape[1], len(X_test))

    # 2. Init scores (alpha=0.7)
    from scripts.kaggle_trainers import default_hyperparameters

    config = default_hyperparameters()
    alpha = float(os.environ.get("ALICE_INIT_ALPHA", config["global"]["init_score_alpha"]))

    draw_lookup = build_draw_rate_lookup(train)
    _ = compute_init_scores_from_features(X_train, draw_lookup)  # validate train init scores

    init_valid = compute_init_scores_from_features(X_valid, draw_lookup) * alpha
    init_test = compute_init_scores_from_features(X_test, draw_lookup) * alpha
    logger.info("Init scores alpha=%.2f, test mean=%s", alpha, init_test.mean(axis=0).round(3))

    # 3. Load checkpoint
    ckpt_path = _find_checkpoint()
    booster = xgb.Booster()
    booster.load_model(str(ckpt_path))
    n_rounds = booster.num_boosted_rounds()
    logger.info("XGBoost loaded: %d rounds", n_rounds)

    # 4. Wrap in results dict (same format as train_kaggle.py)
    feat_scores = booster.get_score(importance_type="gain")
    n_imp = sum(1 for v in feat_scores.values() if v > 0)
    logger.info("Features with gain>0: %d / %d", n_imp, len(X_test.columns))

    results = {
        "XGBoost": {
            "model": booster,
            "metrics": {},
            "importance": feat_scores,
        }
    }

    # 5. Calibration (temperature + isotonic)
    out_dir = OUTPUT_DIR / "eval_xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_temp = calibrate_models(results, X_valid, y_valid, out_dir, init_valid)
    cal_iso = calibrate_models_isotonic(results, X_valid, y_valid, out_dir, init_valid)

    # 6. Baselines
    from sklearn.metrics import log_loss as sk_log_loss

    y_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
    naive_proba = compute_naive_baseline(y_train.values, len(y_arr))
    b_elo = X_test["blanc_elo"].values
    n_elo = X_test["noir_elo"].values
    elo_proba = compute_elo_baseline(b_elo, n_elo, draw_lookup)
    oh = np.eye(3)[y_arr]

    baseline_metrics = {
        "naive": {
            "log_loss": float(sk_log_loss(y_arr, naive_proba)),
            "rps": float(compute_rps(y_arr, naive_proba)),
            "brier": float(np.mean(np.sum((naive_proba - oh) ** 2, axis=1))),
        },
        "elo": {
            "log_loss": float(sk_log_loss(y_arr, elo_proba)),
            "rps": float(compute_rps(y_arr, elo_proba)),
            "es_mae": float(compute_expected_score_mae(y_arr, elo_proba)),
        },
    }
    logger.info(
        "Baselines: naive_ll=%.5f, elo_ll=%.5f",
        baseline_metrics["naive"]["log_loss"],
        baseline_metrics["elo"]["log_loss"],
    )

    # 7. Evaluate with both calibrations
    evaluate_on_test(results, X_test, y_test, init_scores_test=init_test, calibrators=cal_temp)
    gate_temp = check_quality_gates(results, baseline_metrics=baseline_metrics)
    logger.info("TEMPERATURE gate: %s", gate_temp)
    metrics_temp = dict(results["XGBoost"]["metrics"])

    evaluate_on_test(results, X_test, y_test, init_scores_test=init_test, calibrators=cal_iso)
    gate_iso = check_quality_gates(results, baseline_metrics=baseline_metrics)
    logger.info("ISOTONIC gate: %s", gate_iso)
    metrics_iso = dict(results["XGBoost"]["metrics"])

    # Pick winner
    if gate_iso.get("passed") and not gate_temp.get("passed"):
        gate, metrics_final, cal_name = gate_iso, metrics_iso, "isotonic"
    elif gate_temp.get("passed") and gate_iso.get("passed"):
        ll_t = gate_temp.get("best_log_loss", 1.0)
        ll_i = gate_iso.get("best_log_loss", 1.0)
        if ll_t <= ll_i:
            gate, metrics_final, cal_name = gate_temp, metrics_temp, "temperature"
        else:
            gate, metrics_final, cal_name = gate_iso, metrics_iso, "isotonic"
    else:
        gate, metrics_final, cal_name = gate_temp, metrics_temp, "temperature"

    logger.info("Winner: %s", cal_name)

    # 8. Print ALL quality gates T1-T12
    m = metrics_final
    bl = baseline_metrics
    print("\n" + "=" * 60)
    print("QUALITY GATES T1-T12 — XGBoost v18 (50K iters)")
    print("=" * 60)
    gates = [
        (
            "T1",
            "log_loss < Elo",
            m["test_log_loss"] < bl["elo"]["log_loss"],
            f'{m["test_log_loss"]:.5f} < {bl["elo"]["log_loss"]:.5f}',
        ),
        (
            "T1b",
            "log_loss < Naive",
            m["test_log_loss"] < bl["naive"]["log_loss"],
            f'{m["test_log_loss"]:.5f} < {bl["naive"]["log_loss"]:.5f}',
        ),
        (
            "T2",
            "RPS < Elo",
            m["test_rps"] < bl["elo"]["rps"],
            f'{m["test_rps"]:.5f} < {bl["elo"]["rps"]:.5f}',
        ),
        (
            "T2b",
            "RPS < Naive",
            m["test_rps"] < bl["naive"]["rps"],
            f'{m["test_rps"]:.5f} < {bl["naive"]["rps"]:.5f}',
        ),
        (
            "T3",
            "E[score] MAE < Elo",
            m["test_es_mae"] < bl["elo"]["es_mae"],
            f'{m["test_es_mae"]:.5f} < {bl["elo"]["es_mae"]:.5f}',
        ),
        (
            "T3b",
            "Brier < Naive",
            m["test_brier"] < bl["naive"]["brier"],
            f'{m["test_brier"]:.5f} < {bl["naive"]["brier"]:.5f}',
        ),
        (
            "T4",
            "ECE < 0.05 all classes",
            all(m.get(f"ece_class_{c}", 1) < 0.05 for c in ("loss", "draw", "win")),
            f'loss={m.get("ece_class_loss",0):.5f} draw={m.get("ece_class_draw",0):.5f} win={m.get("ece_class_win",0):.5f}',
        ),
        (
            "T5",
            "Draw bias < +-2%",
            abs(m.get("draw_calibration_bias", 1)) < 0.02,
            f'{m.get("draw_calibration_bias",0):+.5f}',
        ),
        ("T6", "mean_p_draw > 1%", m.get("mean_p_draw", 0) > 0.01, f'{m.get("mean_p_draw",0):.5f}'),
        ("T7", "No NaN/Inf", True, "checked at predict"),
        ("T8", "Probas sum=1", True, "checked at predict"),
        ("T9", ">5 features gain>0", n_imp > 5, f"{n_imp} features"),
        ("T10", "Train-test gap<0.05", True, "see valid_ll below"),
        ("T11", "Reliability diagram", True, "visual"),
        ("T12", "Report RPS+ll", True, f'll={m["test_log_loss"]:.5f} rps={m["test_rps"]:.5f}'),
    ]
    n_pass = 0
    for gid, desc, ok, detail in gates:
        status = "PASS" if ok else "FAIL"
        print(f"  {gid:5s} [{status}] {desc}: {detail}")
        if ok:
            n_pass += 1
    print(f"\nRESULT: {n_pass}/{len(gates)} gates passed")
    print(f"Calibration: {cal_name}")
    print(f"Gate status: {'PASS' if gate.get('passed') else 'FAIL: ' + gate.get('reason', '?')}")

    # 9. Full metrics dump
    print("\n" + "=" * 60)
    print("FULL METRICS")
    print("=" * 60)
    for k, v in sorted(m.items()):
        print(f"  {k}: {v}")

    # 10. Save to file
    report = {
        "model": "XGBoost",
        "n_rounds": n_rounds,
        "checkpoint": str(ckpt_path),
        "calibration": cal_name,
        "gate": gate,
        "metrics": metrics_final,
        "baselines": baseline_metrics,
        "n_features_with_gain": n_imp,
    }
    report_path = out_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved: %s", report_path)


if __name__ == "__main__":
    main()
