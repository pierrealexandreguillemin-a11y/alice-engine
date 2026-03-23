"""Kaggle Cloud Training — ALICE Engine V8 MultiClass (ISO 42001/5259/5055).

Kernel 2 of 2: trains CatBoost/XGBoost/LightGBM on pre-computed features.
Requires Kernel 1 (fe_kaggle.py) output via kernel_sources.
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
    import numpy as np  # noqa: PLC0415
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


def _compute_init_scores(X: pd.DataFrame, draw_lookup: pd.DataFrame) -> np.ndarray:
    """Compute Elo baseline init scores from X features."""
    from scripts.baselines import compute_elo_baseline, compute_elo_init_scores  # noqa: PLC0415

    b_elo = X["blanc_elo"].values if "blanc_elo" in X.columns else np.full(len(X), 1500)
    n_elo = X["noir_elo"].values if "noir_elo" in X.columns else np.full(len(X), 1500)
    elo_proba = compute_elo_baseline(b_elo, n_elo, draw_lookup)
    return compute_elo_init_scores(elo_proba)


def main() -> None:
    """Full Kaggle training pipeline orchestration (ISO 42001)."""
    logger.info("ALICE Engine — V8 MultiClass Training (Kernel 2/2)")
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
    from scripts.kaggle_trainers import (  # noqa: PLC0415
        LABEL_COLUMN,
        MODEL_EXTENSIONS,
        check_quality_gates,
        default_hyperparameters,
        evaluate_on_test,
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

    # Feature subset selection (incremental validation: top 10 → all)
    feature_subset = os.environ.get("ALICE_FEATURE_SUBSET", "top10")  # v6: top10 only
    if feature_subset == "top10":
        # Top 11 from CatBoost v3 importance (all features with importance > 0)
        top10 = [
            "diff_elo",
            "elo_proximity",
            "win_rate_home_dom",
            "expected_score_recent_blanc",
            "expected_score_recent_noir",
            "draw_rate_blanc",
            "draw_rate_noir",
            "win_rate_normal_blanc",
            "draw_rate_home_dom",
            "win_rate_normal_noir",
            "ffe_nb_equipes_blanc",
        ]
        keep = [c for c in top10 if c in X_train.columns]
        logger.info("Feature subset: top10 (%d/%d available)", len(keep), len(top10))
        X_train, X_valid, X_test = X_train[keep], X_valid[keep], X_test[keep]

    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    config = default_hyperparameters()
    # CatBoost: write training logs inside versioned out_dir (not cwd)
    config["catboost"]["train_dir"] = str(out_dir / "catboost_info")

    # Compute Elo baseline init scores for residual learning
    from scripts.features.draw_priors import build_draw_rate_lookup  # noqa: PLC0415

    draw_lookup_train = build_draw_rate_lookup(train)
    init_scores_train = _compute_init_scores(X_train, draw_lookup_train)
    init_scores_valid = _compute_init_scores(X_valid, draw_lookup_train)
    init_scores_test = _compute_init_scores(X_test, draw_lookup_train)
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
    )
    evaluate_on_test(results, X_test, y_test, init_scores_test=init_scores_test)

    baseline_metrics, draw_lookup = _compute_baselines(train, X_test, y_test, y_train)
    draw_lookup.to_parquet(out_dir / "draw_rate_lookup.parquet", index=False)
    logger.info("Saved draw_rate_lookup.parquet (%d cells) for inference", len(draw_lookup))

    champion_ll = fetch_champion_ll()
    gate = check_quality_gates(results, baseline_metrics=baseline_metrics, champion_ll=champion_ll)
    logger.info("Quality gate: %s", gate)

    save_models(results, encoders, out_dir, model_extensions=MODEL_EXTENSIONS)
    save_diagnostics(results, X_test, y_test, X_valid, y_valid, X_train, out_dir)
    metadata = build_model_card(results, lineage, gate, config, MODEL_EXTENSIONS, out_dir=out_dir)
    metadata["version"] = version
    if gate.get("passed"):
        save_metadata_and_push(metadata, out_dir)
    else:
        logger.error("Quality gate FAILED: %s — saving locally only, NO push.", gate.get("reason"))
        import json  # noqa: PLC0415

        with open(out_dir / "metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)
    logger.info(
        "Done. Status=%s Best=%s LogLoss=%.4f",
        metadata["status"],
        gate.get("best_model"),
        gate.get("best_log_loss", 0),
    )


if __name__ == "__main__":
    main()
