"""Script: run_iso42001_postprocessing.py - AIMMS Runner (ISO 42001/5055)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from scripts.aimms.aimms_types import AIMSConfig
from scripts.aimms.postprocessor import run_postprocessing
from scripts.training.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main() -> None:
    """Execute AIMMS post-training pipeline (ISO 42001 Clause 8.2/9.1)."""
    model_path, predictor = None, None
    for p in sorted(Path("models/autogluon").glob("autogluon_*"), reverse=True):
        try:
            pred = TabularPredictor.load(str(p))
            if len(pred.leaderboard(silent=True)) > 0:
                model_path, predictor = p, pred
                break
        except Exception as e:  # noqa: BLE001 - Multiple exception types possible
            logger.debug("Skipping %s: %s", p.name, e)
            continue
    if not model_path or predictor is None:
        raise RuntimeError("No valid AutoGluon model found in models/autogluon/")
    valid_df = pd.read_parquet("data/features/valid.parquet")
    test_df = pd.read_parquet("data/features/test.parquet")
    for df in [valid_df, test_df]:
        df["target"] = (df["resultat_blanc"] == 1.0).astype(int)
    result = run_postprocessing(
        model=predictor,
        X_calib=valid_df[FEATURES],
        y_calib=valid_df["target"].values,
        X_test=test_df[FEATURES],
        y_test=test_df["target"].values,
        model_version=model_path.name,
        config=AIMSConfig(calibration_cv=0, uncertainty_alpha=0.10),
    )
    Path("reports").mkdir(exist_ok=True)
    Path("reports/iso42001_aimms.json").write_text(json.dumps(result.to_dict(), indent=2))
    logger.info("Generated: reports/iso42001_aimms.json")
    for rec in result.recommendations:
        logger.info("  → %s", rec)


if __name__ == "__main__":
    main()
