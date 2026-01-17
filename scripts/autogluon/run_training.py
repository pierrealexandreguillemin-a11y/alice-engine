"""Script: run_training.py - AutoGluon Training Runner.

Document ID: ALICE-SCRIPT-AG-TRAIN-001
Version: 1.0.0

Runner script pour Phase 3 du plan ML ISO.
Utilise le module trainer.py existant.

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (<50 lignes)
- ISO/IEC 42001:2023 - AI Management System

Author: ALICE Engine Team
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
    "blanc_elo", "noir_elo", "diff_elo", "echiquier", "niveau", "ronde",
    "type_competition", "division", "ligue_code", "blanc_titre",
    "noir_titre", "jour_semaine",
]


def main() -> None:
    """Execute AutoGluon training pipeline."""
    train_df = pd.read_parquet("data/features/train.parquet")
    valid_df = pd.read_parquet("data/features/valid.parquet")
    test_df = pd.read_parquet("data/features/test.parquet")

    for df in [train_df, valid_df, test_df]:
        df["target"] = (df["resultat_blanc"] == 1.0).astype(int)

    combined = pd.concat([train_df, valid_df], ignore_index=True)
    train_data = combined[FEATURES + ["target"]]
    test_data = test_df[FEATURES + ["target"]]

    config = load_autogluon_config()
    result = train_autogluon(train_data, label="target", config=config)

    test_auc = roc_auc_score(
        test_data["target"],
        result.predictor.predict_proba(test_data.drop(columns="target"))[1],
    )

    report = {
        "test_auc": test_auc,
        "best_model": result.best_model,
        "num_models": result.metrics["num_models"],
        "leaderboard": result.leaderboard[["model", "score_val"]].to_dict("records"),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/autogluon_results.json").write_text(json.dumps(report, indent=2))
    logger.info(f"Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
