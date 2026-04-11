"""Grid search — gap-filling round 2.

Round 1 findings (2026-04-12):
- LGB alpha monotone through 0.3 (gradient -0.008/step). Floor unknown.
- XGB depth=6 optimal but sub/col/mcw optimized for depth=8.

This kernel runs BOTH models sequentially: LGB alpha extension + XGB depth=6 refinement.
~18 min total on 62K train.

ISO: 42001 (traceability), 5259-2 (same temporal subset), 5055 (<300 lines).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
ELO_BASELINE = 0.9766
MAX_RUNTIME_S = 36000
HP_SEARCH_MIN_SEASON = int(os.environ.get("ALICE_HP_MIN_SEASON", "2022"))

# ─── LGB alpha extension: find the floor below 0.3 ──────────────────────────
# Round 1: 0.3 >> 0.4 (delta 0.008). Monotone. Test 0.1/0.15/0.2.
# Other params fixed at Grid v2 best (alpha=0.4 optimized).
LGB_GRID = {"init_score_alpha": [0.1, 0.15, 0.2]}
LGB_FIXED = {
    "num_leaves": 15,
    "feature_fraction": 1.0,
    "min_child_samples": 275,
    "reg_lambda": 4.0,
    "max_depth": 8,
    "learning_rate": 0.05,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "seed": SEED,
    "verbose": -1,
    "n_jobs": int(os.environ.get("ALICE_NTHREAD", "4")),
}

# ─── XGB depth=6 refinement: verify sub/col/mcw at new depth ────────────────
# Round 1: depth=6 + alpha=0.5 = 0.5178 (best). But sub/col/mcw from depth=8 Grid v4.
# depth=6 = 64 leaves max (vs 256 at depth=8). Params may need adjustment.
XGB_GRID = {
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.75, 1.0],
    "min_child_weight": [50, 100],
}
XGB_FIXED = {
    "init_score_alpha": 0.5,
    "max_depth": 6,
    "eta": 0.05,
    "lambda": 4.0,
    "alpha": 0.01,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "seed": SEED,
    "verbosity": 0,
    "nthread": int(os.environ.get("ALICE_NTHREAD", "4")),
}


def _grid_combos(grid: dict) -> list[dict]:
    """Generate all param combinations."""
    keys = list(grid.keys())
    return [dict(zip(keys, vals, strict=False)) for vals in product(*grid.values())]


def _append_csv(row: dict, path: Path) -> None:
    """Append one row to CSV (incremental, survives timeout)."""
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _save_best(results: list[dict], model: str, out_dir: Path) -> None:
    """Save best params JSON after each combo."""
    if not results:
        return
    best = min(results, key=lambda r: r["logloss"])
    skip = {"model", "combo_idx", "logloss", "best_iter", "duration_s"}
    params = {k: v for k, v in best.items() if k not in skip}
    out = {
        "model": model,
        "grid_type": "gap_filling_r2",
        "best_logloss": float(best["logloss"]),
        "best_params": params,
        "n_combos": len(results),
        "total_time_s": round(sum(r["duration_s"] for r in results), 1),
        "elo_baseline_logloss": ELO_BASELINE,
    }
    (out_dir / f"grid_gaps2_{model}.json").write_text(json.dumps(out, indent=2, default=float))


def run_lgb_alpha_extension(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
    out_dir: Path,
) -> list[dict]:
    """LGB alpha={0.1, 0.15, 0.2} — find floor below 0.3."""
    import lightgbm as lgb

    combos = _grid_combos(LGB_GRID)
    csv_path = out_dir / "grid_gaps2_lightgbm.csv"
    logger.info("LGB alpha extension: %d combos", len(combos))
    results: list[dict] = []

    for i, combo in enumerate(combos):
        alpha = combo["init_score_alpha"]
        params = {**LGB_FIXED}
        dtrain = lgb.Dataset(X_train, label=y_train, init_score=(init_train * alpha).ravel())
        dvalid = lgb.Dataset(
            X_valid, label=y_valid, reference=dtrain, init_score=(init_valid * alpha).ravel()
        )
        t0 = time.time()
        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=50000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
        )
        dur = time.time() - t0
        row = {
            "model": "lightgbm",
            "combo_idx": i,
            "logloss": bst.best_score["valid"]["multi_logloss"],
            "best_iter": bst.best_iteration,
            "duration_s": round(dur, 1),
            **combo,
        }
        results.append(row)
        _append_csv(row, csv_path)
        _save_best(results, "lightgbm", out_dir)
        del bst, dtrain, dvalid
        gc.collect()
        logger.info(
            "LGB %d/%d: alpha=%.2f -> %.5f (%.0fs)", i + 1, len(combos), alpha, row["logloss"], dur
        )
    return results


def run_xgb_depth6_refinement(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
    out_dir: Path,
) -> list[dict]:
    """XGB depth=6 fixed, vary sub/col/mcw (8 combos)."""
    import xgboost as xgb

    combos = _grid_combos(XGB_GRID)
    csv_path = out_dir / "grid_gaps2_xgboost.csv"
    logger.info("XGB depth=6 refinement: %d combos", len(combos))
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    results: list[dict] = []

    for i, combo in enumerate(combos):
        params = {**XGB_FIXED, **combo}
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_base_margin((init_train * XGB_FIXED["init_score_alpha"]).ravel())
        dvalid.set_base_margin((init_valid * XGB_FIXED["init_score_alpha"]).ravel())
        t0 = time.time()
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=50000,
            evals=[(dvalid, "valid")],
            callbacks=[xgb.callback.EarlyStopping(rounds=200, save_best=True)],
            verbose_eval=False,
        )
        dur = time.time() - t0
        row = {
            "model": "xgboost",
            "combo_idx": i,
            "logloss": bst.best_score,
            "best_iter": bst.best_iteration,
            "duration_s": round(dur, 1),
            **combo,
        }
        results.append(row)
        _append_csv(row, csv_path)
        _save_best(results, "xgboost", out_dir)
        del bst, dtrain
        gc.collect()
        logger.info(
            "XGB %d/%d: sub=%.1f col=%.2f mcw=%d -> %.5f (%.0fs)",
            i + 1,
            len(combos),
            combo["subsample"],
            combo["colsample_bytree"],
            combo["min_child_weight"],
            row["logloss"],
            dur,
        )
    return results


def main() -> None:
    """Run both LGB alpha extension and XGB depth=6 refinement."""
    from scripts.baselines import compute_init_scores_from_features
    from scripts.cloud.train_kaggle import _load_features, _setup_kaggle_imports
    from scripts.features.draw_priors import build_draw_rate_lookup
    from scripts.kaggle_trainers import prepare_features

    _setup_kaggle_imports()

    out_dir = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw, valid_raw, test_raw, _ = _load_features()
    if HP_SEARCH_MIN_SEASON > 0 and "saison" in train_raw.columns:
        n_before = len(train_raw)
        train_raw = train_raw[train_raw["saison"] >= HP_SEARCH_MIN_SEASON].copy()
        logger.info(
            "HP filter: saison >= %d -- %d -> %d rows",
            HP_SEARCH_MIN_SEASON,
            n_before,
            len(train_raw),
        )

    X_train, y_train, X_valid, y_valid, _, _, _ = prepare_features(train_raw, valid_raw, test_raw)
    logger.info("Features: train=%s, valid=%s", X_train.shape, X_valid.shape)

    for name, df in [("train", X_train), ("valid", X_valid)]:
        dead = [c for c in df.columns if df[c].isna().mean() > 0.99]
        if dead:
            raise ValueError(f"{len(dead)} features >99% NaN on {name}: {dead[:5]}")
    logger.info("NaN audit: PASS")

    draw_lookup = build_draw_rate_lookup(train_raw)
    init_train = compute_init_scores_from_features(X_train, draw_lookup)
    init_valid = compute_init_scores_from_features(X_valid, draw_lookup)
    logger.info("Init scores: train=%s, valid=%s", init_train.shape, init_valid.shape)

    logger.info("=" * 60)
    logger.info("GAP-FILLING ROUND 2: LGB alpha extension + XGB depth=6")
    logger.info("=" * 60)

    run_lgb_alpha_extension(X_train, y_train, init_train, X_valid, y_valid, init_valid, out_dir)
    run_xgb_depth6_refinement(X_train, y_train, init_train, X_valid, y_valid, init_valid, out_dir)


if __name__ == "__main__":
    main()
