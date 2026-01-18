#!/usr/bin/env python3
"""Runner séquentiel des baseline models isolés.

Entraîne les trois baselines et génère un rapport de comparaison.
RÉUTILISE scripts/training (ISO 5055 - pas de duplication).

Usage:
    python -m scripts.baseline.run_baselines
    python -m scripts.baseline.run_baselines --compare-autogluon

Author: ALICE Engine Team
Version: 1.1.0
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent.parent


def main() -> None:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Run Baseline Models (ISO 24029)")
    parser.add_argument("--compare-autogluon", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BASELINE MODELS - Sequential Training")
    logger.info("Reusing scripts/training (ISO 5055)")
    logger.info("=" * 60)

    results: list[BaselineMetrics] = []

    logger.info("\n[1/3] CatBoost...")
    results.append(train_catboost_baseline())

    logger.info("\n[2/3] XGBoost...")
    results.append(train_xgboost_baseline())

    logger.info("\n[3/3] LightGBM...")
    results.append(train_lightgbm_baseline())

    # Rapport
    _generate_report(results, args.compare_autogluon)


def _generate_report(results: list[BaselineMetrics], compare_ag: bool) -> None:
    """Génère le rapport de comparaison."""
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
