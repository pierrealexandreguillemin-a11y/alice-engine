"""Grid search on 200K subsample — one model per kernel.

109 combos × ~330s/combo ≈ 10h on CPU (fits 12h session).
Each combo saved to CSV immediately — no data loss on timeout.

References
----------
- Bergstra & Bengio 2012: ranking stable between subsample and full data
- Probst et al. 2019: 2-3 params drive most tunability for GBDTs
- Benchmarked 2026-04-09: 320s/combo on 200K rows XGBoost CPU
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
MAX_RUNTIME_S = 36000  # 10h — 2h margin for post-processing on 12h Kaggle
_GLOBAL_START: float = 0  # Set in main(), checked in each grid loop

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


def _append_result(row: dict, csv_path: Path) -> None:
    """Append one result row to CSV. Creates file with header if first row."""
    df_row = pd.DataFrame([row])
    df_row.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


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
    csv_path: Path | None = None,
) -> list[dict]:
    """Run XGBoost grid search on subsample."""
    import xgboost as xgb

    combos = _grid_combos(SHARED_GRID, XGBOOST_GRID)
    combos.insert(0, V8_XGBOOST)
    logger.info("XGBoost: %d combos (each ~330s on 200K rows = ~10h total)", len(combos))

    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    results = []

    for i, combo in enumerate(combos):
        if time.time() - _GLOBAL_START > MAX_RUNTIME_S:
            logger.info(
                "TIME LIMIT %ds reached at combo %d/%d — stopping", MAX_RUNTIME_S, i, len(combos)
            )
            break
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
        if csv_path:
            _append_result(row, csv_path)
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
    csv_path: Path | None = None,
) -> list[dict]:
    """Run CatBoost grid search on subsample."""
    from catboost import CatBoostClassifier, Pool

    combos = _grid_combos(SHARED_GRID, CATBOOST_GRID)
    combos.insert(0, V8_CATBOOST)
    logger.info("CatBoost: %d combos (each ~400s on 200K rows = ~12h total)", len(combos))

    results = []
    for i, combo in enumerate(combos):
        if time.time() - _GLOBAL_START > MAX_RUNTIME_S:
            logger.info(
                "TIME LIMIT %ds reached at combo %d/%d — stopping", MAX_RUNTIME_S, i, len(combos)
            )
            break
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
        if csv_path:
            _append_result(row, csv_path)
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
    csv_path: Path | None = None,
) -> list[dict]:
    """Run LightGBM grid search on subsample."""
    import lightgbm as lgb

    combos = _grid_combos(SHARED_GRID, LIGHTGBM_GRID)
    combos.insert(0, V8_LIGHTGBM)
    logger.info("LightGBM: %d combos (each ~250s on 200K rows = ~8h total)", len(combos))

    results = []
    for i, combo in enumerate(combos):
        if time.time() - _GLOBAL_START > MAX_RUNTIME_S:
            logger.info(
                "TIME LIMIT %ds reached at combo %d/%d — stopping", MAX_RUNTIME_S, i, len(combos)
            )
            break
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
        if csv_path:
            _append_result(row, csv_path)
        del booster, dtrain, dvalid
        gc.collect()

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                "LightGBM %d/%d: logloss=%.5f (%.0fs)", i + 1, len(combos), row["logloss"], elapsed
            )

    return results


def main() -> None:
    """Run grid search for one model on 200K subsample.

    Model selected via ALICE_MODEL env var (xgboost/catboost/lightgbm).
    ~109 combos × ~330s/combo ≈ 10h on CPU. One model per 12h kernel.
    """
    from scripts.baselines import compute_init_scores_from_features
    from scripts.cloud.train_kaggle import _load_features, _setup_kaggle_imports
    from scripts.features.draw_priors import build_draw_rate_lookup
    from scripts.kaggle_trainers import prepare_features

    _setup_kaggle_imports()

    global _GLOBAL_START  # noqa: PLW0603
    _GLOBAL_START = time.time()

    model_name = os.environ.get("ALICE_MODEL", "xgboost")
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

    grid_fn = {
        "xgboost": run_xgboost_grid,
        "catboost": run_catboost_grid,
        "lightgbm": run_lightgbm_grid,
    }
    if model_name not in grid_fn:
        msg = f"Unknown model: {model_name}. Expected: {list(grid_fn)}"
        raise ValueError(msg)

    csv_path = out_dir / f"grid_search_{model_name}.csv"
    logger.info("=" * 60)
    logger.info(
        "%s GRID SEARCH — results saved to %s after EACH combo", model_name.upper(), csv_path
    )
    logger.info("=" * 60)
    all_results = grid_fn[model_name](
        X_train,
        y_train,
        init_train,
        X_valid,
        y_valid,
        init_valid,
        csv_path=csv_path,
    )

    # Save results
    df = pd.DataFrame(all_results)
    results_path = out_dir / f"grid_search_{model_name}.csv"
    df.to_csv(results_path, index=False)
    logger.info("Saved %d results to %s", len(df), results_path)

    # Report best
    df_sorted = df.sort_values("logloss")
    best = df_sorted.iloc[0]
    v8_rows = df[df["is_v8_baseline"]]
    v8_logloss = float(v8_rows.iloc[0]["logloss"]) if len(v8_rows) > 0 else float("nan")

    logger.info("=" * 60)
    logger.info(
        "%s: BEST=%.5f | V8=%.5f | delta=%.5f",
        model_name,
        best["logloss"],
        v8_logloss,
        best["logloss"] - v8_logloss,
    )
    param_keys = [
        k
        for k in best.index
        if k not in ["model", "combo_idx", "logloss", "best_iter", "duration_s", "is_v8_baseline"]
    ]
    for rank, (_, row) in enumerate(df_sorted.head(5).iterrows()):
        params = {k: row[k] for k in param_keys}
        logger.info(
            "  #%d: %.5f %s%s",
            rank + 1,
            row["logloss"],
            params,
            " (V8)" if row["is_v8_baseline"] else "",
        )

    best_out = {
        "model": model_name,
        "best_logloss": float(best["logloss"]),
        "best_params": {
            k: float(best[k]) if isinstance(best[k], np.floating) else best[k] for k in param_keys
        },
        "best_iter": int(best["best_iter"]),
        "v8_logloss": v8_logloss,
        "n_combos": len(df),
        "total_time_s": float(df["duration_s"].sum()),
    }
    best_path = out_dir / f"grid_best_{model_name}.json"
    best_path.write_text(json.dumps(best_out, indent=2))
    logger.info("Saved best params to %s", best_path)

    total_time = df["duration_s"].sum()
    logger.info(
        "TOTAL: %d combos in %.0f min (%.1f h)", len(df), total_time / 60, total_time / 3600
    )
