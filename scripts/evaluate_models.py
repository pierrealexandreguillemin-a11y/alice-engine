#!/usr/bin/env python3
"""Module: evaluate_models.py - Evaluation comparative ML.

Evaluation comparative CatBoost vs XGBoost vs LightGBM pour ALICE.
Identifie le meilleur candidat pour ALI.

ISO Compliance:
- ISO/IEC 25059:2023 - AI Quality Model (metriques, benchmarks)
- ISO/IEC 29119 - Software Testing (tests modeles)
- ISO/IEC 42001:2023 - AI Management (evaluation tracable)

Metriques evaluees:
- AUC-ROC (qualite des probabilites)
- Accuracy (precision globale)
- Temps d'entrainement
- Temps d'inference

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --sample 100000

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts.evaluation.constants import DEFAULT_DATA_DIR
from scripts.evaluation.pipeline import run_evaluation

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    """Point d'entree."""
    parser = argparse.ArgumentParser(description="Evaluation ML ALICE")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Repertoire des donnees features",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sous-echantillon pour tests rapides",
    )
    args = parser.parse_args()

    run_evaluation(args.data_dir, args.sample)


if __name__ == "__main__":
    main()
