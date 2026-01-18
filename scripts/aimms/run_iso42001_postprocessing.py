"""Script: run_iso42001_postprocessing.py - AIMMS Runner (ISO 42001/5055 <50 lignes)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from scripts.aimms.aimms_types import AIMSConfig
from scripts.aimms.postprocessor import run_postprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = [
    "blanc_elo", "noir_elo", "diff_elo", "echiquier", "niveau", "ronde",
    "type_competition", "division", "ligue_code", "blanc_titre", "noir_titre", "jour_semaine",
]


def main() -> None:
    """Execute AIMMS post-training pipeline (ISO 42001 Clause 8.2/9.1)."""
    model_path = sorted(Path("models/autogluon").glob("autogluon_*"))[-1]
    predictor = TabularPredictor.load(str(model_path))
    valid_df = pd.read_parquet("data/features/valid.parquet")
    test_df = pd.read_parquet("data/features/test.parquet")
    for df in [valid_df, test_df]:
        df["target"] = (df["resultat_blanc"] == 1.0).astype(int)

    result = run_postprocessing(
        model=predictor, X_calib=valid_df[FEATURES], y_calib=valid_df["target"].values,
        X_test=test_df[FEATURES], y_test=test_df["target"].values,
        model_version=model_path.name, config=AIMSConfig(calibration_cv=0, uncertainty_alpha=0.10),
    )
    Path("reports").mkdir(exist_ok=True)
    Path("reports/iso42001_aimms.json").write_text(json.dumps(result.to_dict(), indent=2))
    logger.info("Generated: reports/iso42001_aimms.json")
    for rec in result.recommendations:
        logger.info(f"  → {rec}")


if __name__ == "__main__":
    main()
