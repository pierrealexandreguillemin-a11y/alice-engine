"""Grid search on 200K subsample — the pragmatic approach.

108 combos × 3 models × ~30s/combo = ~3-4h total on CPU.
Validates Optuna V9 findings with exhaustive coverage.

References
----------
- Bergstra & Bengio 2012: ranking stable between subsample and full data
- Probst et al. 2019: 2-3 params drive most tunability for GBDTs
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

SUBSAMPLE = 200_000
SEED = 42
ELO_BASELINE = 0.9766

# Grid — 4 params, same as Optuna V9 reduced search space
SHARED_GRID = {
    "init_score_alpha": [0.5, 0.6, 0.7, 0.8],
}

XGBOOST_GRID = {
    "subsample": [0.6, 0.7, 0.8],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "min_child_weight": [50, 100, 200],
}
# Fixed XGBoost params (fANOVA + literature)
XGBOOST_FIXED = {
    "max_depth": 8,
    "eta": 0.05,
    "lambda": 4.0,
    "alpha": 0.01,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "seed": 42,
    "verbosity": 0,
    "nthread": int(os.environ.get("ALICE_NTHREAD", "4")),
}

CATBOOST_GRID = {
    "depth": [4, 6, 8, 10],
    "rsm": [0.3, 0.5, 0.7],
    "min_data_in_leaf": [50, 100, 200],
}
CATBOOST_FIXED = {
    "learning_rate": 0.05,
    "l2_leaf_reg": 4.0,
    "random_strength": 2.0,
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 50000,
    "early_stopping_rounds": 200,
    "task_type": "CPU",
    "random_seed": 42,
    "verbose": 0,
}

LIGHTGBM_GRID = {
    "num_leaves": [31, 63, 127, 255],
    "feature_fraction": [0.3, 0.6, 0.9],
    "bagging_fraction": [0.5, 0.7, 0.9],
}
LIGHTGBM_FIXED = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "reg_lambda": 4.0,
    "min_child_samples": 100,
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
    "n_jobs": int(os.environ.get("ALICE_NTHREAD", "4")),
}

# V8 champion params (manual tuning, 18 iterations) — baseline reference
V8_XGBOOST = {
    "init_score_alpha": 0.7,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "min_child_weight": 50,
}
V8_CATBOOST = {"init_score_alpha": 0.7, "depth": 4, "rsm": 0.3, "min_data_in_leaf": 200}
V8_LIGHTGBM = {
    "init_score_alpha": 0.7,
    "num_leaves": 15,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.7,
}


def _grid_combos(shared: dict, model_grid: dict) -> list[dict]:
    """Generate all combinations of shared + model-specific params."""
    all_keys = list(shared.keys()) + list(model_grid.keys())
    all_values = list(shared.values()) + list(model_grid.values())
    combos = []
    for vals in product(*all_values):
        combos.append(dict(zip(all_keys, vals, strict=False)))
    return combos


def run_xgboost_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
) -> list[dict]:
    """Run XGBoost grid search on subsample."""
    import xgboost as xgb

    combos = _grid_combos(SHARED_GRID, XGBOOST_GRID)
    # Add V8 baseline as first combo
    combos.insert(0, V8_XGBOOST)
    logger.info("XGBoost: %d combos", len(combos))

    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    results = []

    for i, combo in enumerate(combos):
        alpha = combo["init_score_alpha"]
        params = {**XGBOOST_FIXED}
        params["subsample"] = combo["subsample"]
        params["colsample_bytree"] = combo["colsample_bytree"]
        params["min_child_weight"] = combo["min_child_weight"]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_base_margin((init_train * alpha).ravel())
        dvalid.set_base_margin((init_valid * alpha).ravel())

        t0 = time.time()
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=50000,
            evals=[(dvalid, "valid")],
            callbacks=[xgb.callback.EarlyStopping(rounds=200, save_best=True)],
            verbose_eval=False,
        )
        elapsed = time.time() - t0

        row = {
            "model": "xgboost",
            "combo_idx": i,
            "logloss": booster.best_score,
            "best_iter": booster.best_iteration,
            "duration_s": round(elapsed, 1),
            "is_v8_baseline": (i == 0),
            **combo,
        }
        results.append(row)
        del booster, dtrain
        gc.collect()

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                "XGBoost %d/%d: logloss=%.5f (%.0fs)", i + 1, len(combos), row["logloss"], elapsed
            )

    return results


def run_catboost_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
) -> list[dict]:
    """Run CatBoost grid search on subsample."""
    from catboost import CatBoostClassifier, Pool

    combos = _grid_combos(SHARED_GRID, CATBOOST_GRID)
    combos.insert(0, V8_CATBOOST)
    logger.info("CatBoost: %d combos", len(combos))

    results = []
    for i, combo in enumerate(combos):
        alpha = combo["init_score_alpha"]
        params = {**CATBOOST_FIXED}
        params["depth"] = combo["depth"]
        params["rsm"] = combo["rsm"]
        params["min_data_in_leaf"] = combo["min_data_in_leaf"]

        train_pool = Pool(X_train, y_train, baseline=(init_train * alpha))
        valid_pool = Pool(X_valid, y_valid, baseline=(init_valid * alpha))

        t0 = time.time()
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool)
        elapsed = time.time() - t0

        val_scores = model.get_best_score().get("validation", {})
        logloss = val_scores.get(
            "MultiClass", val_scores.get(next(iter(val_scores), ""), float("inf"))
        )

        row = {
            "model": "catboost",
            "combo_idx": i,
            "logloss": logloss,
            "best_iter": model.get_best_iteration(),
            "duration_s": round(elapsed, 1),
            "is_v8_baseline": (i == 0),
            **combo,
        }
        results.append(row)
        del model, train_pool, valid_pool
        gc.collect()

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                "CatBoost %d/%d: logloss=%.5f (%.0fs)", i + 1, len(combos), row["logloss"], elapsed
            )

    return results


def run_lightgbm_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
) -> list[dict]:
    """Run LightGBM grid search on subsample."""
    import lightgbm as lgb

    combos = _grid_combos(SHARED_GRID, LIGHTGBM_GRID)
    combos.insert(0, V8_LIGHTGBM)
    logger.info("LightGBM: %d combos", len(combos))

    results = []
    for i, combo in enumerate(combos):
        alpha = combo["init_score_alpha"]
        num_leaves = min(combo["num_leaves"], 2 ** LIGHTGBM_FIXED["max_depth"] - 1)
        params = {**LIGHTGBM_FIXED, "num_leaves": num_leaves}
        params["feature_fraction"] = combo["feature_fraction"]
        params["bagging_fraction"] = combo["bagging_fraction"]

        dtrain = lgb.Dataset(X_train, label=y_train, init_score=(init_train * alpha).ravel())
        dvalid = lgb.Dataset(
            X_valid, label=y_valid, reference=dtrain, init_score=(init_valid * alpha).ravel()
        )

        t0 = time.time()
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=50000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
        )
        elapsed = time.time() - t0

        logloss = booster.best_score["valid"]["multi_logloss"]
        row = {
            "model": "lightgbm",
            "combo_idx": i,
            "logloss": logloss,
            "best_iter": booster.best_iteration,
            "duration_s": round(elapsed, 1),
            "is_v8_baseline": (i == 0),
            **combo,
        }
        results.append(row)
        del booster, dtrain, dvalid
        gc.collect()

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                "LightGBM %d/%d: logloss=%.5f (%.0fs)", i + 1, len(combos), row["logloss"], elapsed
            )

    return results


def main() -> None:
    """Run grid search for all 3 models on 200K subsample."""
    from scripts.baselines import compute_init_scores_from_features
    from scripts.cloud.train_kaggle import _load_features, _setup_kaggle_imports
    from scripts.features.draw_priors import build_draw_rate_lookup
    from scripts.kaggle_trainers import prepare_features

    _setup_kaggle_imports()

    out_dir = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    train_raw, valid_raw, test_raw, _ = _load_features()
    logger.info("Raw: train=%d, valid=%d", len(train_raw), len(valid_raw))

    # Subsample train (valid stays full for stable eval)
    if len(train_raw) > SUBSAMPLE:
        train_raw = train_raw.sample(SUBSAMPLE, random_state=SEED)
        logger.info("Subsampled train to %d rows", SUBSAMPLE)

    X_train, y_train, X_valid, y_valid, _, _, _ = prepare_features(train_raw, valid_raw, test_raw)
    logger.info("Features: train=%s, valid=%s", X_train.shape, X_valid.shape)

    draw_lookup = build_draw_rate_lookup(train_raw)
    init_train = compute_init_scores_from_features(X_train, draw_lookup)
    init_valid = compute_init_scores_from_features(X_valid, draw_lookup)
    logger.info("Init scores ready")

    # Run all 3 grids
    all_results = []

    logger.info("=" * 60)
    logger.info("XGBOOST GRID SEARCH")
    logger.info("=" * 60)
    all_results.extend(run_xgboost_grid(X_train, y_train, init_train, X_valid, y_valid, init_valid))

    logger.info("=" * 60)
    logger.info("CATBOOST GRID SEARCH")
    logger.info("=" * 60)
    all_results.extend(
        run_catboost_grid(X_train, y_train, init_train, X_valid, y_valid, init_valid)
    )

    logger.info("=" * 60)
    logger.info("LIGHTGBM GRID SEARCH")
    logger.info("=" * 60)
    all_results.extend(
        run_lightgbm_grid(X_train, y_train, init_train, X_valid, y_valid, init_valid)
    )

    # Save all results
    df = pd.DataFrame(all_results)
    results_path = out_dir / "grid_search_results.csv"
    df.to_csv(results_path, index=False)
    logger.info("Saved %d results to %s", len(df), results_path)

    # Report best per model
    logger.info("=" * 60)
    logger.info("BEST PARAMS PER MODEL")
    logger.info("=" * 60)
    best_per_model = {}
    for model_name in ["xgboost", "catboost", "lightgbm"]:
        model_df = df[df["model"] == model_name].sort_values("logloss")
        best = model_df.iloc[0]
        v8 = model_df[model_df["is_v8_baseline"]].iloc[0]
        logger.info(
            "%s: BEST=%.5f (combo %d) | V8=%.5f | delta=%.5f",
            model_name,
            best["logloss"],
            best["combo_idx"],
            v8["logloss"],
            best["logloss"] - v8["logloss"],
        )
        # Log top 5
        for rank, (_, row) in enumerate(model_df.head(5).iterrows()):
            params = {
                k: row[k]
                for k in row.index
                if k
                not in [
                    "model",
                    "combo_idx",
                    "logloss",
                    "best_iter",
                    "duration_s",
                    "is_v8_baseline",
                ]
            }
            logger.info(
                "  #%d: %.5f %s%s",
                rank + 1,
                row["logloss"],
                params,
                " (V8)" if row["is_v8_baseline"] else "",
            )

        best_per_model[model_name] = {
            "best_logloss": float(best["logloss"]),
            "best_params": {
                k: best[k]
                for k in best.index
                if k
                not in [
                    "model",
                    "combo_idx",
                    "logloss",
                    "best_iter",
                    "duration_s",
                    "is_v8_baseline",
                ]
            },
            "best_iter": int(best["best_iter"]),
            "v8_logloss": float(v8["logloss"]),
            "n_combos": len(model_df),
            "total_time_s": float(model_df["duration_s"].sum()),
        }

    # Save best params
    best_path = out_dir / "grid_search_best_params.json"
    best_path.write_text(json.dumps(best_per_model, indent=2))
    logger.info("Saved best params to %s", best_path)

    # Summary
    total_time = df["duration_s"].sum()
    logger.info("=" * 60)
    logger.info(
        "TOTAL: %d combos in %.0f min (%.1f h)", len(df), total_time / 60, total_time / 3600
    )
    logger.info("=" * 60)
