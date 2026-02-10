#!/usr/bin/env python3
"""Runner sequentiel des baseline models isoles.

Entraine les trois baselines et genere un rapport de comparaison.
REUTILISE scripts/training (ISO 5055 - pas de duplication).

Usage:
    python -m scripts.baseline.run_baselines
    python -m scripts.baseline.run_baselines --compare-autogluon

Author: ALICE Engine Team
Version: 1.2.0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scripts.baseline import (
    BaselineMetrics,
    train_catboost_baseline,
    train_lightgbm_baseline,
    train_xgboost_baseline,
)
from scripts.fairness.auto_report import (
    format_markdown_report,
    generate_comprehensive_report,
)
from scripts.fairness.protected import validate_features
from scripts.fairness.protected.config import DEFAULT_PROTECTED_ATTRIBUTES
from scripts.training.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent.parent
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _run_protected_attrs_check() -> None:
    """Valide les features contre les attributs proteges (ISO 24027)."""
    import pandas as pd

    train_path = PROJECT_DIR / "data" / "features" / "train.parquet"
    df = pd.read_parquet(train_path) if train_path.exists() else None
    result = validate_features(
        ALL_FEATURES,
        df=df,
        categorical_features=CATEGORICAL_FEATURES,
    )
    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "protected_attrs_baselines.json").write_text(
        json.dumps(result.to_dict(), indent=2)
    )
    if not result.is_valid:
        msg = f"Protected attributes violation: {result.violations}"
        raise ValueError(msg)
    logger.info("Protected attrs check: %d warnings", len(result.warnings))


def main() -> None:
    """Point d'entree principal."""
    parser = argparse.ArgumentParser(description="Run Baseline Models (ISO 24029)")
    parser.add_argument("--compare-autogluon", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BASELINE MODELS - Sequential Training")
    logger.info("Reusing scripts/training (ISO 5055)")
    logger.info("=" * 60)

    _run_protected_attrs_check()

    results: list[BaselineMetrics] = []

    logger.info("\n[1/3] CatBoost...")
    results.append(train_catboost_baseline())

    logger.info("\n[2/3] XGBoost...")
    results.append(train_xgboost_baseline())

    logger.info("\n[3/3] LightGBM...")
    results.append(train_lightgbm_baseline())

    _generate_report(results, args.compare_autogluon)
    _run_fairness_reports()


def _run_fairness_reports() -> None:
    """Genere les rapports fairness pour les baselines (ISO 24027)."""
    import numpy as np
    import pandas as pd

    test_path = PROJECT_DIR / "data" / "features" / "test.parquet"
    if not test_path.exists():
        logger.warning("Test data not found, skipping fairness reports")
        return
    test_df = pd.read_parquet(test_path)
    y_true = (test_df["resultat_blanc"] == 1.0).astype(int).values

    for model_name in ["catboost", "xgboost", "lightgbm"]:
        pred_path = PROJECT_DIR / "models" / "baseline" / f"{model_name}_predictions.npy"
        if not pred_path.exists():
            logger.warning("No predictions for %s, skipping fairness report", model_name)
            continue
        y_pred = np.load(pred_path)
        report = generate_comprehensive_report(
            y_true,
            y_pred,
            test_df,
            model_name,
            "baseline",
            DEFAULT_PROTECTED_ATTRIBUTES,
        )
        reports_dir = PROJECT_DIR / "reports"
        reports_dir.mkdir(exist_ok=True)
        (reports_dir / f"fairness_{model_name}.json").write_text(
            json.dumps(report.to_dict(), indent=2)
        )
        format_markdown_report(report, output_path=reports_dir / f"fairness_{model_name}.md")
    logger.info("Fairness reports generated for baselines")


def _generate_report(results: list[BaselineMetrics], compare_ag: bool) -> None:
    """Genere le rapport de comparaison."""
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE COMPARISON REPORT")
    logger.info("=" * 60)

    best_auc, best_model = 0.0, ""
    for r in results:
        logger.info(f"\n{r.model_name}: AUC={r.auc_roc:.4f} | Time={r.train_time_s:.1f}s")
        if r.auc_roc > best_auc:
            best_auc, best_model = r.auc_roc, r.model_name

    logger.info(f"\nBest baseline: {best_model} (AUC: {best_auc:.4f})")

    if compare_ag:
        ag_path = PROJECT_DIR / "reports" / "autogluon_results.json"
        if ag_path.exists():
            with open(ag_path) as f:
                ag = json.load(f)
            ag_auc = ag.get("test_auc", ag.get("validation_auc", 0))
            logger.info(f"\nAutoGluon AUC: {ag_auc:.4f}")
            for r in results:
                diff = r.auc_roc - ag_auc
                logger.info(f"  {r.model_name} vs AG: {diff:+.4f}")

    # Sauvegarder
    output = PROJECT_DIR / "models" / "baseline" / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(
            {
                "baselines": [{"model": r.model_name, "auc": r.auc_roc} for r in results],
                "best": best_model,
            },
            f,
            indent=2,
        )
    logger.info(f"\nReport: {output}")


if __name__ == "__main__":
    main()
