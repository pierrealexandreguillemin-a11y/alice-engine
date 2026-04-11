"""Grid search — gap-filling for Training Final decision.

Targets the specific parameter gaps identified in V9 HP search audit:
- P0: alpha < 0.5 never tested for XGB and CB, alpha < 0.4 never tested for LGB
- P1: depth never compared for XGB (always fixed at 8)
- P1: lambda never explored for LGB (always fixed at 4.0)

Same dataset as all V9 HP search: saison=2022 (~62K train, ~71K valid).
Same init_scores, same feature pipeline. Results comparable to Grid v2/v4 and Optuna.

ISO Compliance:
- ISO/IEC 42001:2023 — traceability (CSV + JSON per combo, incremental save)
- ISO/IEC 5259-2:2024 — same temporal subset as all V9 experiments
- ISO/IEC 5055:2021 — SRP, <300 lines
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

# ─── XGBoost gap grid: alpha × depth (9 combos + V8 baseline) ───────────────
# Gap P0: alpha < 0.5 never tested. Gap P1: depth 4 vs 6 vs 8 never compared.
# Other params fixed at best V9 Grid v4: sub=0.8, col=1.0, mcw=50
XGB_GAP_GRID = {
    "init_score_alpha": [0.3, 0.5, 0.7],
    "max_depth": [4, 6, 8],
}
XGB_GAP_FIXED = {
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 50,
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
XGB_V8 = {"init_score_alpha": 0.7, "max_depth": 4}

# ─── LightGBM gap grid: alpha × lambda (16 combos + V8 baseline) ────────────
# Gap P0: alpha < 0.4 never tested. Gap P1: lambda never explored for LGB.
# Other params fixed at best V9 Grid v2: leaves=15, ff=1.0, mcs=275
LGB_GAP_GRID = {
    "init_score_alpha": [0.3, 0.4, 0.5, 0.7],
    "reg_lambda": [1.0, 4.0, 10.0, 15.0],
}
LGB_GAP_FIXED = {
    "num_leaves": 15,
    "feature_fraction": 1.0,
    "min_child_samples": 275,
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
LGB_V8 = {"init_score_alpha": 0.7, "reg_lambda": 10.0}

# ─── CatBoost gap grid: alpha × depth (16 combos + V8 baseline) ─────────────
# Gap P0: alpha < 0.5 never tested. Gap P1: depth {4,5,6,8} to refine Grid v2.
# Other params fixed at best V9 Grid v2: l2=8, rsm=0.7
CB_GAP_GRID = {
    "init_score_alpha": [0.3, 0.4, 0.5, 0.7],
    "depth": [4, 5, 6, 8],
}
CB_GAP_FIXED = {
    "l2_leaf_reg": 8.0,
    "rsm": 0.7,
    "learning_rate": 0.05,
    "random_strength": 2.0,
    "min_data_in_leaf": 200,
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 50000,
    "early_stopping_rounds": 200,
    "task_type": "CPU",
    "random_seed": SEED,
    "verbose": 0,
}
CB_V8 = {"init_score_alpha": 0.7, "depth": 4}


def _grid_combos(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals, strict=False)) for vals in product(*grid.values())]


def _append_csv(row: dict, path: Path) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _save_best(results: list[dict], model: str, out_dir: Path) -> None:
    if not results:
        return
    best = min(results, key=lambda r: r["logloss"])
    v8 = [r for r in results if r.get("is_v8_baseline")]
    param_keys = [
        k
        for k in best
        if k not in ("model", "combo_idx", "logloss", "best_iter", "duration_s", "is_v8_baseline")
    ]
    out = {
        "model": model,
        "grid_type": "gap_filling",
        "best_logloss": float(best["logloss"]),
        "best_params": {k: best[k] for k in param_keys},
        "v8_logloss": float(v8[0]["logloss"]) if v8 else None,
        "n_combos": len(results),
        "total_time_s": round(sum(r["duration_s"] for r in results), 1),
        "elo_baseline_logloss": ELO_BASELINE,
    }
    (out_dir / f"grid_gaps_{model}.json").write_text(json.dumps(out, indent=2, default=float))


def run_xgboost_gaps(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
    out_dir: Path,
) -> list[dict]:
    """XGBoost gap grid: alpha={0.3,0.5,0.7} x depth={4,6,8}."""
    import xgboost as xgb

    combos = [XGB_V8] + _grid_combos(XGB_GAP_GRID)
    csv_path = out_dir / "grid_gaps_xgboost.csv"
    logger.info("XGBoost gaps: %d combos (V8 baseline + %d grid)", len(combos), len(combos) - 1)

    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    results: list[dict] = []
    t_global = time.time()

    for i, combo in enumerate(combos):
        if time.time() - t_global > MAX_RUNTIME_S:
            logger.warning("TIME LIMIT at combo %d/%d", i, len(combos))
            break

        alpha = combo["init_score_alpha"]
        depth = combo["max_depth"]
        params = {**XGB_GAP_FIXED, "max_depth": depth}

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_base_margin((init_train * alpha).ravel())
        dvalid.set_base_margin((init_valid * alpha).ravel())

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
            "is_v8_baseline": (i == 0),
            **combo,
        }
        results.append(row)
        _append_csv(row, csv_path)
        _save_best(results, "xgboost", out_dir)
        del bst, dtrain
        gc.collect()
        logger.info(
            "XGB %d/%d: alpha=%.1f depth=%d → %.5f (%.0fs)",
            i + 1,
            len(combos),
            alpha,
            depth,
            row["logloss"],
            dur,
        )

    return results


def run_lightgbm_gaps(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
    out_dir: Path,
) -> list[dict]:
    """LightGBM gap grid: alpha={0.3,0.4,0.5,0.7} x lambda={1,4,10,15}."""
    import lightgbm as lgb

    combos = [LGB_V8] + _grid_combos(LGB_GAP_GRID)
    csv_path = out_dir / "grid_gaps_lightgbm.csv"
    logger.info("LightGBM gaps: %d combos (V8 baseline + %d grid)", len(combos), len(combos) - 1)

    results: list[dict] = []
    t_global = time.time()

    for i, combo in enumerate(combos):
        if time.time() - t_global > MAX_RUNTIME_S:
            logger.warning("TIME LIMIT at combo %d/%d", i, len(combos))
            break

        alpha = combo["init_score_alpha"]
        lam = combo["reg_lambda"]
        params = {**LGB_GAP_FIXED, "reg_lambda": lam}

        dtrain = lgb.Dataset(X_train, label=y_train, init_score=(init_train * alpha).ravel())
        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=dtrain,
            init_score=(init_valid * alpha).ravel(),
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
            "is_v8_baseline": (i == 0),
            **combo,
        }
        results.append(row)
        _append_csv(row, csv_path)
        _save_best(results, "lightgbm", out_dir)
        del bst, dtrain, dvalid
        gc.collect()
        logger.info(
            "LGB %d/%d: alpha=%.1f lambda=%.1f → %.5f (%.0fs)",
            i + 1,
            len(combos),
            alpha,
            lam,
            row["logloss"],
            dur,
        )

    return results


def run_catboost_gaps(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
    out_dir: Path,
) -> list[dict]:
    """CatBoost gap grid: alpha={0.3,0.4,0.5,0.7} x depth={4,5,6,8}."""
    from catboost import CatBoostClassifier, Pool

    combos = [CB_V8] + _grid_combos(CB_GAP_GRID)
    csv_path = out_dir / "grid_gaps_catboost.csv"
    logger.info("CatBoost gaps: %d combos (V8 baseline + %d grid)", len(combos), len(combos) - 1)

    results: list[dict] = []
    t_global = time.time()

    for i, combo in enumerate(combos):
        if time.time() - t_global > MAX_RUNTIME_S:
            logger.warning("TIME LIMIT at combo %d/%d", i, len(combos))
            break

        alpha = combo["init_score_alpha"]
        depth = combo["depth"]
        params = {**CB_GAP_FIXED, "depth": depth}

        train_pool = Pool(X_train, y_train, baseline=(init_train * alpha))
        valid_pool = Pool(X_valid, y_valid, baseline=(init_valid * alpha))

        t0 = time.time()
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool)
        dur = time.time() - t0

        val_scores = model.get_best_score().get("validation", {})
        logloss = val_scores.get("MultiClass", float("inf"))

        row = {
            "model": "catboost",
            "combo_idx": i,
            "logloss": logloss,
            "best_iter": model.get_best_iteration(),
            "duration_s": round(dur, 1),
            "is_v8_baseline": (i == 0),
            **combo,
        }
        results.append(row)
        _append_csv(row, csv_path)
        _save_best(results, "catboost", out_dir)
        del model, train_pool, valid_pool
        gc.collect()
        logger.info(
            "CB %d/%d: alpha=%.1f depth=%d → %.5f (%.0fs)",
            i + 1,
            len(combos),
            alpha,
            depth,
            row["logloss"],
            dur,
        )

    return results


def main() -> None:
    """Run gap-filling grid for one model (ALICE_MODEL env var)."""
    from scripts.baselines import compute_init_scores_from_features
    from scripts.cloud.train_kaggle import _load_features, _setup_kaggle_imports
    from scripts.features.draw_priors import build_draw_rate_lookup
    from scripts.kaggle_trainers import prepare_features

    _setup_kaggle_imports()

    model_name = os.environ.get("ALICE_MODEL", "xgboost")
    out_dir = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw, valid_raw, test_raw, _ = _load_features()
    if HP_SEARCH_MIN_SEASON > 0 and "saison" in train_raw.columns:
        n_before = len(train_raw)
        train_raw = train_raw[train_raw["saison"] >= HP_SEARCH_MIN_SEASON].copy()
        logger.info(
            "HP filter: saison >= %d — %d → %d rows", HP_SEARCH_MIN_SEASON, n_before, len(train_raw)
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

    runners = {
        "xgboost": run_xgboost_gaps,
        "lightgbm": run_lightgbm_gaps,
        "catboost": run_catboost_gaps,
    }
    if model_name not in runners:
        raise ValueError(f"Unknown model: {model_name}")

    logger.info("=" * 60)
    logger.info("GAP-FILLING GRID: %s", model_name.upper())
    logger.info("=" * 60)
    runners[model_name](X_train, y_train, init_train, X_valid, y_valid, init_valid, out_dir)


if __name__ == "__main__":
    main()
