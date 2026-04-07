"""Stacking ensemble evaluation — ISO 25059/24029.

Document ID: ALICE-STACKING-EVAL
Version: 1.0.0

Evaluates LogisticRegression and MLP meta-learners on 3 converged V8
models' predictions. No retraining — uses existing prediction parquets.

References
----------
- scikit-learn StackingClassifier concepts
- Karaaslan & Erbay (2025), MDPI Electronics 15(1):1
- Phase 2 serving design: docs/superpowers/specs/2026-04-07-phase2-serving-design.md
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier

from scripts.kaggle_metrics import (
    compute_ece,
    compute_expected_score_mae,
    compute_multiclass_brier,
    compute_rps,
)

logger = logging.getLogger(__name__)

_RAW_COLS = ["y_proba_loss", "y_proba_draw", "y_proba_win"]
_CAL_COLS = ["y_proba_cal_loss", "y_proba_cal_draw", "y_proba_cal_win"]


def assemble_meta_features(
    pred_dfs: list[Any],
    calibrated: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (n, 9) meta-feature matrix from 3 models' predictions.

    Args:
    ----
        pred_dfs: List of 3 DataFrames with prediction columns.
        calibrated: Use calibrated probas (True) or raw (False).

    Returns:
    -------
        (X, y): X is (n, 9) float64, y is (n,) int64.

    Raises:
    ------
        ValueError: If y_true differs across models.
    """
    cols = _CAL_COLS if calibrated else _RAW_COLS
    y_ref = pred_dfs[0]["y_true"].values

    arrays = []
    for i, df in enumerate(pred_dfs):
        if not np.array_equal(df["y_true"].values, y_ref):
            msg = f"y_true mismatch between model 0 and model {i}"
            raise ValueError(msg)
        arrays.append(df[cols].values.astype(np.float64))

    X = np.hstack(arrays)
    return X, y_ref.astype(np.int64)


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute all quality gate metrics for a probability matrix.

    IMPORTANT: compute_ece takes POSITIONAL args (y_true_binary, y_proba_class).
    """
    y_pred = y_proba.argmax(axis=1)

    return {
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1, 2]),
        "rps": compute_rps(y_true, y_proba),
        "es_mae": compute_expected_score_mae(y_true, y_proba),
        "brier": compute_multiclass_brier(y_true, y_proba),
        "ece_loss": compute_ece((y_true == 0).astype(float), y_proba[:, 0]),
        "ece_draw": compute_ece((y_true == 1).astype(float), y_proba[:, 1]),
        "ece_win": compute_ece((y_true == 2).astype(float), y_proba[:, 2]),
        "draw_calibration_bias": float(y_proba[:, 1].mean() - (y_true == 1).astype(float).mean()),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def fit_meta_learner(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    kind: str = "lr",
) -> tuple[Any, np.ndarray]:
    """Fit a meta-learner on stacked predictions, return model + test probas."""
    if kind == "lr":
        model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
        )
    elif kind == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(16,),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
    else:
        msg = f"Unknown meta-learner kind: {kind}"
        raise ValueError(msg)

    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)
    return model, probas


def calibrate_probas(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Post-hoc isotonic calibration of a fitted meta-learner."""
    calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(X_train, y_train)
    return calibrator.predict_proba(X_test)


def main() -> None:
    """Run full stacking evaluation on V8 prediction artifacts."""
    import json
    from pathlib import Path

    import pandas as pd

    xgb_dir = Path("reports/v8_xgboost_v5_resume")
    lgb_dir = Path("C:/Users/pierr/Downloads/lightgbm-v7-output/v20260405_172602")
    cb_dir = Path("C:/Users/pierr/Downloads/catboost-v6-output/v20260405_081528")

    splits: dict[str, list[pd.DataFrame]] = {}
    for split in ("valid", "test"):
        splits[split] = [
            pd.read_parquet(xgb_dir / f"XGBoost_{split}_predictions.parquet"),
            pd.read_parquet(lgb_dir / f"LightGBM_{split}_predictions.parquet"),
            pd.read_parquet(cb_dir / f"CatBoost_{split}_predictions.parquet"),
        ]
        logger.info("Loaded %s: %d rows", split, len(splits[split][0]))

    X_valid, y_valid = assemble_meta_features(splits["valid"], calibrated=True)
    X_test, y_test = assemble_meta_features(splits["test"], calibrated=True)
    logger.info("Meta-features: valid=%s, test=%s", X_valid.shape, X_test.shape)

    model_names = ["XGBoost_v5", "LightGBM_v7", "CatBoost_v6"]
    results: dict[str, dict[str, float]] = {}

    for i, name in enumerate(model_names):
        probas_test = splits["test"][i][_CAL_COLS].values.astype(np.float64)
        results[name] = compute_all_metrics(y_test, probas_test)
        logger.info(
            "%s: log_loss=%.5f, es_mae=%.5f",
            name,
            results[name]["log_loss"],
            results[name]["es_mae"],
        )

    xgb_test = splits["test"][0][_CAL_COLS].values.astype(np.float64)
    lgb_test = splits["test"][1][_CAL_COLS].values.astype(np.float64)
    cb_test = splits["test"][2][_CAL_COLS].values.astype(np.float64)
    blend_90_5_5 = 0.90 * xgb_test + 0.05 * lgb_test + 0.05 * cb_test
    results["Blend_90_5_5"] = compute_all_metrics(y_test, blend_90_5_5)

    lr_model, lr_probas = fit_meta_learner(X_valid, y_valid, X_test, kind="lr")
    results["Stack_LR"] = compute_all_metrics(y_test, lr_probas)
    logger.info(
        "Stack_LR: log_loss=%.5f, es_mae=%.5f",
        results["Stack_LR"]["log_loss"],
        results["Stack_LR"]["es_mae"],
    )

    lr_cal_probas = calibrate_probas(lr_model, X_valid, y_valid, X_test)
    results["Stack_LR_cal"] = compute_all_metrics(y_test, lr_cal_probas)

    mlp_model, mlp_probas = fit_meta_learner(X_valid, y_valid, X_test, kind="mlp")
    results["Stack_MLP"] = compute_all_metrics(y_test, mlp_probas)
    logger.info(
        "Stack_MLP: log_loss=%.5f, es_mae=%.5f",
        results["Stack_MLP"]["log_loss"],
        results["Stack_MLP"]["es_mae"],
    )

    mlp_cal_probas = calibrate_probas(mlp_model, X_valid, y_valid, X_test)
    results["Stack_MLP_cal"] = compute_all_metrics(y_test, mlp_cal_probas)

    X_valid_raw, _ = assemble_meta_features(splits["valid"], calibrated=False)
    X_test_raw, _ = assemble_meta_features(splits["test"], calibrated=False)
    _, lr_raw_probas = fit_meta_learner(X_valid_raw, y_valid, X_test_raw, kind="lr")
    results["Stack_LR_raw"] = compute_all_metrics(y_test, lr_raw_probas)

    print("\n" + "=" * 100)
    print("STACKING EVALUATION RESULTS — V8 MultiClass (test set, 231,532 samples)")
    print("=" * 100)
    header = (
        f"{'Method':<20} {'log_loss':>10} {'rps':>10}"
        f" {'es_mae':>10} {'brier':>10} {'ece_draw':>10}"
        f" {'draw_bias':>10} {'accuracy':>10} {'f1_macro':>10}"
    )
    print(header)
    print("-" * 100)
    for name, m in results.items():
        row = (
            f"{name:<20} "
            f"{m['log_loss']:>10.5f} "
            f"{m['rps']:>10.5f} "
            f"{m['es_mae']:>10.5f} "
            f"{m['brier']:>10.5f} "
            f"{m['ece_draw']:>10.5f} "
            f"{m['draw_calibration_bias']:>+10.5f} "
            f"{m['accuracy']:>10.4f} "
            f"{m['f1_macro']:>10.4f}"
        )
        print(row)

    xgb_es_mae = results["XGBoost_v5"]["es_mae"]
    best_stack_name = min(
        [k for k in results if k.startswith("Stack")],
        key=lambda k: results[k]["es_mae"],
    )
    best_stack_es_mae = results[best_stack_name]["es_mae"]
    gain = xgb_es_mae - best_stack_es_mae

    print("\n--- DECISION GATE ---")
    print(f"XGBoost v5 E[score] MAE: {xgb_es_mae:.5f}")
    print(f"Best stacking ({best_stack_name}) E[score] MAE: {best_stack_es_mae:.5f}")
    print(f"Gain: {gain:+.5f} ({'SIGNIFICANT' if gain > 0.001 else 'NOT SIGNIFICANT'})")

    if gain > 0.001:
        print(f"\n>>> RECOMMENDATION: Use {best_stack_name} — serve 3 models + meta-learner")
    else:
        print("\n>>> RECOMMENDATION: Use XGBoost v5 alone — stacking adds no meaningful gain")

    out_path = Path("reports/stacking_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: {mk: round(mv, 6) for mk, mv in v.items()} for k, v in results.items()}
    serializable["_decision"] = {
        "xgb_es_mae": round(xgb_es_mae, 6),
        "best_stack": best_stack_name,
        "best_stack_es_mae": round(best_stack_es_mae, 6),
        "gain": round(gain, 6),
        "significant": gain > 0.001,
    }
    out_path.write_text(json.dumps(serializable, indent=2))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    main()
