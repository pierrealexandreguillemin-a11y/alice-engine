"""Script: run_training.py - AutoGluon Training Runner.

Document ID: ALICE-SCRIPT-AG-TRAIN-001
Version: 1.1.0
ISO: 5055 (<50 lignes/fonction), 42001 (tracabilite)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.autogluon.config import load_autogluon_config
from scripts.autogluon.trainer import train_autogluon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = [
    "blanc_elo",
    "noir_elo",
    "diff_elo",
    "echiquier",
    "niveau",
    "ronde",
    "type_competition",
    "division",
    "ligue_code",
    "blanc_titre",
    "noir_titre",
    "jour_semaine",
]


def main() -> None:
    """Execute AutoGluon training pipeline (ISO 42001)."""
    train_df = pd.read_parquet("data/features/train.parquet")
    valid_df = pd.read_parquet("data/features/valid.parquet")
    test_df = pd.read_parquet("data/features/test.parquet")

    for df in [train_df, valid_df, test_df]:
        df["target"] = (df["resultat_blanc"] == 1.0).astype(int)

    combined = pd.concat([train_df, valid_df], ignore_index=True)
    result = train_autogluon(
        combined[FEATURES + ["target"]], label="target", config=load_autogluon_config()
    )

    test_proba = result.predictor.predict_proba(test_df[FEATURES])[1]
    test_auc = roc_auc_score(test_df["target"], test_proba)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/autogluon_results.json").write_text(
        json.dumps(
            {
                "test_auc": test_auc,
                "best_model": result.best_model,
                "num_models": result.metrics["num_models"],
            },
            indent=2,
        )
    )
    logger.info("Test AUC: %.4f", test_auc)


if __name__ == "__main__":
    main()
