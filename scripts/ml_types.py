#!/usr/bin/env python3
"""Types stricts pour le pipeline ML ALICE.

Ce module definit les types et protocols pour garantir un typage strict
sans utilisation de `Any` dans les scripts ML.

Conformite:
- ISO/IEC 5055 (Code Quality)
- PEP 544 (Protocols)
- Ruff ANN401 (no Any)

Usage:
    from scripts.ml_types import MLClassifier, MLConfig, TrainingResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray  # noqa: TC002 - needed at runtime for type aliases

# ==============================================================================
# PROTOCOLS - Interfaces pour les modeles ML
# ==============================================================================


@runtime_checkable
class MLClassifier(Protocol):
    """Protocol pour tout classificateur ML compatible sklearn."""

    def fit(
        self,
        X: pd.DataFrame | NDArray[np.float64],
        y: pd.Series | NDArray[np.int64],
        **kwargs: object,
    ) -> MLClassifier:
        """Entraine le modele."""
        ...

    def predict(
        self,
        X: pd.DataFrame | NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Predit les classes."""
        ...

    def predict_proba(
        self,
        X: pd.DataFrame | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predit les probabilites."""
        ...


@runtime_checkable
class CatBoostModel(Protocol):
    """Protocol specifique CatBoost avec save_model."""

    def fit(
        self,
        X: pd.DataFrame | NDArray[np.float64],
        y: pd.Series | NDArray[np.int64],
        **kwargs: object,
    ) -> CatBoostModel:
        """Entraine le modele."""
        ...

    def predict_proba(
        self,
        X: pd.DataFrame | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predit les probabilites."""
        ...

    def save_model(self, path: str) -> None:
        """Sauvegarde le modele au format .cbm."""
        ...

    def get_params(self) -> dict[str, object]:
        """Retourne les hyperparametres."""
        ...


@runtime_checkable
class XGBoostModel(Protocol):
    """Protocol specifique XGBoost avec save_model."""

    def fit(
        self,
        X: pd.DataFrame | NDArray[np.float64],
        y: pd.Series | NDArray[np.int64],
        **kwargs: object,
    ) -> XGBoostModel:
        """Entraine le modele."""
        ...

    def predict_proba(
        self,
        X: pd.DataFrame | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predit les probabilites."""
        ...

    def save_model(self, path: str) -> None:
        """Sauvegarde le modele au format .ubj."""
        ...

    def get_params(self) -> dict[str, object]:
        """Retourne les hyperparametres."""
        ...


@runtime_checkable
class LightGBMModel(Protocol):
    """Protocol specifique LightGBM avec booster_."""

    def fit(
        self,
        X: pd.DataFrame | NDArray[np.float64],
        y: pd.Series | NDArray[np.int64],
        **kwargs: object,
    ) -> LightGBMModel:
        """Entraine le modele."""
        ...

    def predict_proba(
        self,
        X: pd.DataFrame | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predit les probabilites."""
        ...

    @property
    def booster_(self) -> object:
        """Retourne le booster interne."""
        ...

    def get_params(self) -> dict[str, object]:
        """Retourne les hyperparametres."""
        ...


# ==============================================================================
# TYPED DICTS - Configuration structuree
# ==============================================================================


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


# ==============================================================================
# DATACLASSES - Resultats structures
# ==============================================================================


@dataclass
class ModelMetrics:
    """Metriques d'un modele."""

    auc_roc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    log_loss: float
    train_time_s: float = 0.0
    test_auc: float = 0.0
    test_accuracy: float = 0.0
    test_f1: float = 0.0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Convertit en dictionnaire."""
        return {
            "auc_roc": self.auc_roc,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "log_loss": self.log_loss,
            "train_time_s": self.train_time_s,
            "test_auc": self.test_auc,
            "test_accuracy": self.test_accuracy,
            "test_f1": self.test_f1,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_positives": self.true_positives,
        }


@dataclass
class TrainingResult:
    """Resultat d'entrainement d'un modele."""

    model: MLClassifier | None
    train_time: float
    metrics: ModelMetrics

    @property
    def is_valid(self) -> bool:
        """Verifie si l'entrainement a reussi."""
        return self.model is not None


@dataclass
class StackingResult:
    """Resultat du stacking ensemble."""

    meta_model: MLClassifier
    oof_predictions: NDArray[np.float64]
    test_predictions: NDArray[np.float64]
    stacking_test_proba: NDArray[np.float64]
    model_weights: dict[str, float]
    model_names: list[str]
    single_model_metrics: dict[str, dict[str, float]]
    stacking_metrics: dict[str, float]
    best_single_name: str
    best_single_auc: float
    use_stacking: bool
    n_folds: int


@dataclass
class ModelRegistry:
    """Registre des modeles sauvegardes."""

    version: str
    version_dir: Path
    models: dict[str, Path] = field(default_factory=dict)
    metadata_path: Path | None = None
    encoders_path: Path | None = None


# ==============================================================================
# TYPE ALIASES
# ==============================================================================

# Features et labels
Features = pd.DataFrame
Labels = pd.Series
FeaturesArray = NDArray[np.float64]
LabelsArray = NDArray[np.int64]

# Resultats
ModelResults = dict[str, TrainingResult]

# Noms de modeles
ModelName = str
VALID_MODEL_NAMES: tuple[ModelName, ...] = ("CatBoost", "XGBoost", "LightGBM")
