"""Grid search Tier 2 — production-oriented params with draw calibration metrics.

Every combo evaluated on logloss + ECE draw + draw_bias (not just logloss).
The CE uses E[score] = P(win) + 0.5*P(draw). Draw_bias directly impacts CE decisions.

XGBoost: colsample_bynode, gamma, max_delta_step (never explored)
LightGBM: num_leaves scaling for 1.1M (62K optimal=15, but 76K samples/leaf on 1.1M)

ISO: 42001 (traceability), 5259-2 (temporal subset), 25059 (calibration quality gate)
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
DRAW_CLASS = 1  # loss=0, draw=1, win=2


def _compute_draw_metrics(y_true: np.ndarray, probas: np.ndarray) -> dict:
    """Compute draw-specific calibration metrics for CE production needs."""
    p_draw = probas[:, DRAW_CLASS]
    is_draw = (y_true == DRAW_CLASS).astype(float)
    observed_draw_rate = is_draw.mean()
    mean_p_draw = p_draw.mean()
    draw_bias = mean_p_draw - observed_draw_rate

    # ECE draw (15 bins)
    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_draw = 0.0
    for i in range(n_bins):
        mask = (p_draw >= bin_edges[i]) & (p_draw < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = is_draw[mask].mean()
            bin_conf = p_draw[mask].mean()
            ece_draw += mask.sum() * abs(bin_acc - bin_conf)
    ece_draw /= len(y_true)

    return {
        "draw_bias": round(float(draw_bias), 5),
        "ece_draw": round(float(ece_draw), 5),
        "mean_p_draw": round(float(mean_p_draw), 5),
        "observed_draw_rate": round(float(observed_draw_rate), 5),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ─── XGBoost Tier 2: colsample_bynode × gamma × max_delta_step ──────────────
XGB_GRID = {
    "colsample_bynode": [0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 1.0],
    "max_delta_step": [0, 1, 5],
}
XGB_FIXED = {
    "init_score_alpha": 0.5,
    "max_depth": 6,
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

# ─── LightGBM Tier 2: num_leaves scaling ─────────────────────────────────────
LGB_GRID = {
    "num_leaves": [15, 31, 63, 127],
    "min_gain_to_split": [0, 0.01],
    "lambda_l1": [0, 0.1],
}
LGB_FIXED = {
    "init_score_alpha": 0.1,
    "max_depth": 8,
    "feature_fraction": 1.0,
    "min_child_samples": 275,
    "learning_rate": 0.05,
    "reg_lambda": 4.0,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "seed": SEED,
    "verbose": -1,
    "n_jobs": int(os.environ.get("ALICE_NTHREAD", "4")),
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
    best_draw = min(results, key=lambda r: abs(r["draw_bias"]))
    skip = {
        "model",
        "combo_idx",
        "logloss",
        "best_iter",
        "duration_s",
        "draw_bias",
        "ece_draw",
        "mean_p_draw",
        "observed_draw_rate",
    }
    out = {
        "model": model,
        "grid_type": "tier2_production",
        "best_logloss": {
            "value": best["logloss"],
            "draw_bias": best["draw_bias"],
            "ece_draw": best["ece_draw"],
            "params": {k: v for k, v in best.items() if k not in skip},
        },
        "best_draw_calibration": {
            "draw_bias": best_draw["draw_bias"],
            "ece_draw": best_draw["ece_draw"],
            "logloss": best_draw["logloss"],
            "params": {k: v for k, v in best_draw.items() if k not in skip},
        },
        "n_combos": len(results),
        "total_time_s": round(sum(r["duration_s"] for r in results), 1),
    }
    (out_dir / f"grid_tier2_{model}.json").write_text(json.dumps(out, indent=2, default=float))


def run_xgb_tier2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
    out_dir: Path,
) -> list[dict]:
    """XGBoost Tier 2: colsample_bynode × gamma × max_delta_step (27 combos)."""
    import xgboost as xgb

    combos = _grid_combos(XGB_GRID)
    csv_path = out_dir / "grid_tier2_xgboost.csv"
    logger.info("XGB Tier 2: %d combos", len(combos))

    alpha = XGB_FIXED["init_score_alpha"]
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dvalid.set_base_margin((init_valid * alpha).ravel())
    results: list[dict] = []

    for i, combo in enumerate(combos):
        if time.time() - results[0]["_t0"] > MAX_RUNTIME_S if results else False:
            break
        params = {**XGB_FIXED, **combo}
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_base_margin((init_train * alpha).ravel())

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

        # Predict probas for draw metrics
        raw = bst.predict(dvalid)  # multi:softprob returns probas directly
        draw_metrics = _compute_draw_metrics(y_valid.values, raw)

        row = {
            "model": "xgboost",
            "combo_idx": i,
            "logloss": bst.best_score,
            "best_iter": bst.best_iteration,
            "duration_s": round(dur, 1),
            **draw_metrics,
            **combo,
        }
        if not results:
            row["_t0"] = t0
        results.append(row)
        _append_csv({k: v for k, v in row.items() if k != "_t0"}, csv_path)
        _save_best(
            [{k: v for k, v in r.items() if k != "_t0"} for r in results], "xgboost", out_dir
        )
        del bst, dtrain
        gc.collect()
        logger.info(
            "XGB %d/%d: bynode=%.1f gamma=%.1f mds=%d -> loss=%.5f draw_bias=%+.4f ece_draw=%.4f (%.0fs)",
            i + 1,
            len(combos),
            combo["colsample_bynode"],
            combo["gamma"],
            combo["max_delta_step"],
            row["logloss"],
            row["draw_bias"],
            row["ece_draw"],
            dur,
        )
    return results


def run_lgb_tier2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
    out_dir: Path,
) -> list[dict]:
    """LightGBM Tier 2: num_leaves × min_gain × lambda_l1 (16 combos)."""
    import lightgbm as lgb

    combos = _grid_combos(LGB_GRID)
    csv_path = out_dir / "grid_tier2_lightgbm.csv"
    logger.info("LGB Tier 2: %d combos", len(combos))

    alpha = LGB_FIXED["init_score_alpha"]
    results: list[dict] = []
    t_global = time.time()

    for i, combo in enumerate(combos):
        if time.time() - t_global > MAX_RUNTIME_S:
            break
        leaves = min(combo["num_leaves"], 2 ** LGB_FIXED["max_depth"] - 1)
        params = {
            **LGB_FIXED,
            "num_leaves": leaves,
            "min_gain_to_split": combo["min_gain_to_split"],
            "lambda_l1": combo["lambda_l1"],
        }

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

        # Predict raw + init for draw metrics
        raw = bst.predict(X_valid, raw_score=True)
        final = raw + (init_valid * alpha)
        probas = _softmax(final)
        draw_metrics = _compute_draw_metrics(y_valid.values, probas)

        row = {
            "model": "lightgbm",
            "combo_idx": i,
            "logloss": bst.best_score["valid"]["multi_logloss"],
            "best_iter": bst.best_iteration,
            "duration_s": round(dur, 1),
            **draw_metrics,
            **combo,
        }
        results.append(row)
        _append_csv(row, csv_path)
        _save_best(results, "lightgbm", out_dir)
        del bst, dtrain, dvalid
        gc.collect()
        logger.info(
            "LGB %d/%d: leaves=%d mgs=%.2f l1=%.1f -> loss=%.5f draw_bias=%+.4f ece_draw=%.4f (%.0fs)",
            i + 1,
            len(combos),
            leaves,
            combo["min_gain_to_split"],
            combo["lambda_l1"],
            row["logloss"],
            row["draw_bias"],
            row["ece_draw"],
            dur,
        )
    return results


def main() -> None:
    """Run Tier 2 grids: XGB + LGB with draw calibration metrics."""
    model_name = os.environ.get("ALICE_MODEL", "xgboost")

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
    logger.info("TIER 2 GRID: %s (logloss + ECE draw + draw_bias)", model_name.upper())
    logger.info("=" * 60)

    if model_name == "xgboost":
        run_xgb_tier2(X_train, y_train, init_train, X_valid, y_valid, init_valid, out_dir)
    elif model_name == "lightgbm":
        run_lgb_tier2(X_train, y_train, init_train, X_valid, y_valid, init_valid, out_dir)
    else:
        raise ValueError(f"Tier 2 not implemented for {model_name}")


if __name__ == "__main__":
    main()
