"""Configurations ML TypedDict - ISO 5055.

Ce module definit les TypedDict pour la configuration:
- CatBoostConfig, XGBoostConfig, LightGBMConfig
- StackingConfig, GlobalConfig
- MLConfig (configuration complete)

Conformite:
- ISO/IEC 5055 (Code Quality)
- Ruff ANN401 (no Any)
"""

from __future__ import annotations

from typing import TypedDict


class CatBoostConfig(TypedDict, total=False):
    """Configuration CatBoost."""

    iterations: int
    learning_rate: float
    depth: int
    l2_leaf_reg: float
    early_stopping_rounds: int
    random_seed: int
    verbose: int
    thread_count: int
    task_type: str


class XGBoostConfig(TypedDict, total=False):
    """Configuration XGBoost."""

    n_estimators: int
    learning_rate: float
    max_depth: int
    reg_lambda: float
    reg_alpha: float
    tree_method: str
    early_stopping_rounds: int
    random_state: int
    verbosity: int
    n_jobs: int


class LightGBMConfig(TypedDict, total=False):
    """Configuration LightGBM."""

    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    reg_lambda: float
    reg_alpha: float
    early_stopping_rounds: int
    random_state: int
    verbose: int
    n_jobs: int


class StackingConfig(TypedDict, total=False):
    """Configuration stacking."""

    meta_learner: str
    logistic_regression: dict[str, float | int]
    ridge: dict[str, float]
    selection: dict[str, float]


class GlobalConfig(TypedDict, total=False):
    """Configuration globale."""

    random_seed: int
    n_folds: int
    early_stopping_rounds: int
    eval_metric: str
    task_type: str
    verbose: int


class MLConfig(TypedDict, total=False):
    """Configuration complete ML."""

    global_: GlobalConfig
    catboost: CatBoostConfig
    xgboost: XGBoostConfig
    lightgbm: LightGBMConfig
    stacking: StackingConfig
