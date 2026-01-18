"""Baseline models isolés pour comparaison AutoGluon.

Ce package fournit des wrappers fins autour de scripts/training
pour entraîner les modèles de manière INDÉPENDANTE et les comparer.

ISO Compliance:
- ISO/IEC 24029:2021 - Independent implementations for comparison
- ISO/IEC 5055:2021 - Code Quality (réutilisation, pas de duplication)

Usage:
    python -m scripts.baseline.run_baselines --compare-autogluon

Author: ALICE Engine Team
Version: 1.1.0
"""

from scripts.baseline.catboost_baseline import train_catboost_baseline
from scripts.baseline.lightgbm_baseline import train_lightgbm_baseline
from scripts.baseline.types import BaselineMetrics
from scripts.baseline.xgboost_baseline import train_xgboost_baseline

__all__ = [
    "BaselineMetrics",
    "train_catboost_baseline",
    "train_xgboost_baseline",
    "train_lightgbm_baseline",
]
