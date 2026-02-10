"""Script: run_training.py - AutoGluon Training Runner.

Document ID: ALICE-SCRIPT-AG-TRAIN-001
Version: 1.2.0
ISO: 5055 (<50 lignes/fonction), 42001 (tracabilite), 24027 (bias)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.autogluon.config import load_autogluon_config
from scripts.autogluon.trainer import train_autogluon
from scripts.fairness.auto_report import (
    format_markdown_report,
    generate_comprehensive_report,
)
from scripts.fairness.protected import validate_features
from scripts.fairness.protected.config import DEFAULT_PROTECTED_ATTRIBUTES
from scripts.model_registry.rollback import detect_degradation
from scripts.model_registry.versioning import get_current_version

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


_CATEGORICAL = [
    "type_competition",
    "division",
    "ligue_code",
    "blanc_titre",
    "noir_titre",
    "jour_semaine",
]


def _run_protected_attrs_check(train_df: pd.DataFrame) -> None:
    """Valide les features contre les attributs proteges (ISO 24027)."""
    result = validate_features(
        FEATURES,
        df=train_df,
        categorical_features=_CATEGORICAL,
    )
    Path("reports").mkdir(exist_ok=True)
    Path("reports/protected_attrs_validation.json").write_text(
        json.dumps(result.to_dict(), indent=2)
    )
    if not result.is_valid:
        msg = f"Protected attributes violation: {result.violations}"
        raise ValueError(msg)


def main() -> None:
    """Execute AutoGluon training pipeline (ISO 42001)."""
    train_df = pd.read_parquet("data/features/train.parquet")
    valid_df = pd.read_parquet("data/features/valid.parquet")
    test_df = pd.read_parquet("data/features/test.parquet")

    for df in [train_df, valid_df, test_df]:
        df["target"] = (df["resultat_blanc"] == 1.0).astype(int)

    _run_protected_attrs_check(train_df)

    combined = pd.concat([train_df, valid_df], ignore_index=True)
    result = train_autogluon(
        combined[FEATURES + ["target"]], label="target", config=load_autogluon_config()
    )

    test_proba = result.predictor.predict_proba(test_df[FEATURES])[1]
    test_preds = (test_proba >= 0.5).astype(int)
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
    _run_fairness_report(test_df, test_preds)
    _run_rollback_check()
    logger.info("Test AUC: %.4f", test_auc)


def _run_rollback_check() -> None:
    """Verifie la degradation et logge la decision (ISO 23894)."""
    models_dir = Path("models/production")
    if not models_dir.exists():
        return
    current = get_current_version(models_dir)
    if not current:
        return
    decision = detect_degradation(models_dir, current)
    if decision.should_rollback:
        logger.warning("ROLLBACK RECOMMENDED: %s", decision.reason)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/rollback_decision.json").write_text(json.dumps(decision.to_dict(), indent=2))


def _run_fairness_report(test_df: pd.DataFrame, test_preds: pd.Series) -> None:
    """Genere le rapport fairness apres training (ISO 24027)."""
    y_true = test_df["target"].values
    report = generate_comprehensive_report(
        y_true,
        test_preds,
        test_df,
        "AutoGluon",
        "latest",
        DEFAULT_PROTECTED_ATTRIBUTES,
    )
    Path("reports/fairness_autogluon.json").write_text(json.dumps(report.to_dict(), indent=2))
    format_markdown_report(report, output_path=Path("reports/fairness_autogluon.md"))
    logger.info("Fairness: %s", report.overall_status)


if __name__ == "__main__":
    main()
