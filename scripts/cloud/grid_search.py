"""Grid search on recent season (saison=2022) — one model per kernel.

81 combos (3^4) per model. Each combo saved to CSV immediately.
Same data as Optuna for direct comparison of HP search methods.

HP Search Methodology (v2.0, 2026-04-11):
- Train on saison=2022 (~62K rows), valid on 2023 (~71K rows)
- Rationale: HP directions stable between subset and full data
  (AUTOMATA, Killamsetty et al. NeurIPS 2022)
- Budget: XGB ~147s/combo (3.3h), LGB ~107s/combo (2.4h), CB ~422s/combo (9.5h)
- Training Final uses full dataset (2002-2022) with best params found here

References
----------
- AUTOMATA: https://arxiv.org/abs/2203.08212
- Bergstra & Bengio 2012: ranking stable between subsample and full data
- Benchmarked 2026-04-10: 286s/combo LGB, 392s/combo XGB, 1125s/combo CB on 200K
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
MAX_RUNTIME_S = 36000  # 10h — 2h margin for post-processing on 12h Kaggle
_GLOBAL_START: float = 0  # Set in main(), checked in each grid loop

# HP search on recent season — same as Optuna for direct comparison.
HP_SEARCH_MIN_SEASON = int(os.environ.get("ALICE_HP_MIN_SEASON", "2022"))

# ─── Grids: 3^4 = 81 combos per model (3 levels × 4 params) ───────────────
# Same 4 params as Optuna V9 search space. Alpha always included.
# Levels: endpoints + midpoint of each Optuna range.

XGBOOST_GRID = {
    "init_score_alpha": [0.5, 0.65, 0.8],  # Optuna range [0.5, 0.8]
    "subsample": [0.6, 0.7, 0.8],  # Optuna range [0.6, 0.8]
    "colsample_bytree": [0.5, 0.75, 1.0],  # Optuna range [0.5, 1.0]
    "min_child_weight": [50, 125, 200],  # Optuna range [50, 200]
}
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
    "init_score_alpha": [0.5, 0.65, 0.8],  # Optuna range [0.5, 0.8]
    "depth": [4, 7, 10],  # Optuna range [4, 10]
    "l2_leaf_reg": [
        1.0,
        8.0,
        15.0,
    ],  # Optuna range [1, 15]. Replaces min_data_in_leaf (zero effect oblivious trees)
    "rsm": [0.2, 0.45, 0.7],  # Optuna range [0.2, 0.7]
}
CATBOOST_FIXED = {
    "learning_rate": 0.05,
    "random_strength": 2.0,
    "min_data_in_leaf": 200,  # FIXED — zero effect on oblivious trees
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 50000,
    "early_stopping_rounds": 200,
    "task_type": "CPU",
    "random_seed": 42,
    "verbose": 0,
}

LIGHTGBM_GRID = {
    "init_score_alpha": [0.4, 0.6, 0.8],  # Optuna range [0.4, 0.8]
    "num_leaves": [15, 135, 255],  # Optuna range [15, 255]
    "feature_fraction": [0.3, 0.65, 1.0],  # Optuna range [0.3, 1.0]
    "min_child_samples": [50, 275, 500],  # Optuna range [50, 500]
}
LIGHTGBM_FIXED = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "reg_lambda": 4.0,
    "bagging_fraction": 0.8,  # fANOVA 1.4%, fixed near-optimal
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
    "n_jobs": int(os.environ.get("ALICE_NTHREAD", "4")),
}

# V8 champion params (manual tuning) — baseline reference (combo 0 in each grid)
V8_XGBOOST = {
    "init_score_alpha": 0.7,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "min_child_weight": 50,
}
V8_CATBOOST = {"init_score_alpha": 0.7, "depth": 4, "l2_leaf_reg": 10.0, "rsm": 0.3}
V8_LIGHTGBM = {
    "init_score_alpha": 0.7,
    "num_leaves": 15,
    "feature_fraction": 0.5,
    "min_child_samples": 200,
}


def _append_result(row: dict, csv_path: Path) -> None:
    """Append one result row to CSV. Creates file with header if first row."""
    df_row = pd.DataFrame([row])
    df_row.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def _save_best_json(results: list[dict], model_name: str, out_dir: Path) -> None:
    """Save best params JSON after each combo — survives timeout."""
    if not results:
        return
    best = min(results, key=lambda r: r["logloss"])
    v8_rows = [r for r in results if r.get("is_v8_baseline")]
    v8_logloss = v8_rows[0]["logloss"] if v8_rows else float("nan")
    param_keys = [
        k
        for k in best
        if k not in ("model", "combo_idx", "logloss", "best_iter", "duration_s", "is_v8_baseline")
    ]
    out = {
        "model": model_name,
        "best_logloss": float(best["logloss"]),
        "best_params": {
            k: float(best[k])
            if isinstance(best[k], float | np.floating)
            else int(best[k])
            if isinstance(best[k], int | np.integer)
            else best[k]
            for k in param_keys
        },
        "best_iter": int(best["best_iter"]),
        "v8_logloss": v8_logloss,
        "n_combos": len(results),
        "total_time_s": round(sum(r["duration_s"] for r in results), 1),
    }
    (out_dir / f"grid_best_{model_name}.json").write_text(json.dumps(out, indent=2))


def _grid_combos(grid: dict) -> list[dict]:
    """Generate all combinations from a single grid dict."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, vals, strict=False)) for vals in product(*values)]


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

    combos = _grid_combos(XGBOOST_GRID)
    combos.insert(0, V8_XGBOOST)
    logger.info("XGBoost: %d combos (V8 baseline + %d grid)", len(combos), len(combos) - 1)

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
            _save_best_json(results, "xgboost", csv_path.parent)
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

    combos = _grid_combos(CATBOOST_GRID)
    combos.insert(0, V8_CATBOOST)
    logger.info("CatBoost: %d combos (V8 baseline + %d grid)", len(combos), len(combos) - 1)

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
        params["l2_leaf_reg"] = combo["l2_leaf_reg"]

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
            _save_best_json(results, "catboost", csv_path.parent)
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

    combos = _grid_combos(LIGHTGBM_GRID)
    combos.insert(0, V8_LIGHTGBM)
    logger.info("LightGBM: %d combos (V8 baseline + %d grid)", len(combos), len(combos) - 1)

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
        params["min_child_samples"] = combo["min_child_samples"]

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
            _save_best_json(results, "lightgbm", csv_path.parent)
        del booster, dtrain, dvalid
        gc.collect()

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                "LightGBM %d/%d: logloss=%.5f (%.0fs)", i + 1, len(combos), row["logloss"], elapsed
            )

    return results


def main() -> None:
    """Run grid search for one model on recent season (saison=2022).

    Model selected via ALICE_MODEL env var (xgboost/catboost/lightgbm).
    81 combos (3^4) + V8 baseline. Same data as Optuna for comparison.
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

    # Filter train to recent season (same as Optuna — AUTOMATA NeurIPS 2022, ISO 5259-2)
    if HP_SEARCH_MIN_SEASON > 0 and "saison" in train_raw.columns:
        n_before = len(train_raw)
        train_raw = train_raw[train_raw["saison"] >= HP_SEARCH_MIN_SEASON].copy()
        logger.info(
            "HP search filter: saison >= %d — %d → %d rows (%.1f%%)",
            HP_SEARCH_MIN_SEASON,
            n_before,
            len(train_raw),
            100 * len(train_raw) / n_before,
        )

    X_train, y_train, X_valid, y_valid, _, _, _ = prepare_features(train_raw, valid_raw, test_raw)
    logger.info("Features: train=%s, valid=%s", X_train.shape, X_valid.shape)

    # NaN audit — mandatory before training (2 weeks lost debugging 61 dead features)
    for split_name, df_split in [("train", X_train), ("valid", X_valid)]:
        dead = [c for c in df_split.columns if df_split[c].isna().mean() > 0.99]
        if dead:
            logger.error("STOP: %d features >99%% NaN on %s: %s", len(dead), split_name, dead[:10])
            raise ValueError(f"{len(dead)} features >99% NaN on {split_name}")
    logger.info("NaN audit: PASS (no features >99%% NaN)")

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
            k: float(best[k])
            if isinstance(best[k], np.floating)
            else int(best[k])
            if isinstance(best[k], np.integer)
            else best[k]
            for k in param_keys
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
