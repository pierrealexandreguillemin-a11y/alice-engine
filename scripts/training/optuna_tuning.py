"""Module: scripts/training/optuna_tuning.py - Thin re-export.

Document ID: ALICE-MOD-OPTUNA-001
Version: 2.0.0

DEPRECATED: Ce module est conserve pour compatibilite ascendante.
Importer directement depuis scripts.training.optuna_core.

Le module original a ete refactore en modules SRP (ISO 5055):
- optuna_core.py: optimize_hyperparameters, optimize_all_models
- optuna_objectives.py: create_catboost/xgboost/lightgbm_objective

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (SRP, <30 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from scripts.training.optuna_core import optimize_all_models, optimize_hyperparameters
from scripts.training.optuna_objectives import (
    _create_catboost_objective,
    _create_lightgbm_objective,
    _create_xgboost_objective,
)

__all__ = [
    "optimize_hyperparameters",
    "optimize_all_models",
    "_create_catboost_objective",
    "_create_lightgbm_objective",
    "_create_xgboost_objective",
]
