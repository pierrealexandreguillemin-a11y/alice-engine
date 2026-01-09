"""Package Evaluation - ISO 5055.

Ce package contient les modules pour l'evaluation comparative des modeles:
- constants.py: Constantes et configuration
- data.py: Preparation des features
- trainers.py: Entrainement des modeles (CatBoost, XGBoost, LightGBM)
- metrics.py: Calcul des metriques d'evaluation
- pipeline.py: Pipeline principal d'evaluation
"""

from scripts.evaluation.constants import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATA_DIR,
    DEFAULT_DEPTH,
    DEFAULT_EARLY_STOPPING,
    DEFAULT_ITERATIONS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    NUMERIC_FEATURES,
    PROJECT_DIR,
)
from scripts.evaluation.data import prepare_features
from scripts.evaluation.metrics import evaluate_model
from scripts.evaluation.pipeline import run_evaluation
from scripts.evaluation.trainers import train_catboost, train_lightgbm, train_xgboost

__all__ = [
    # Constants
    "PROJECT_DIR",
    "DEFAULT_DATA_DIR",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "DEFAULT_ITERATIONS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_DEPTH",
    "DEFAULT_EARLY_STOPPING",
    "DEFAULT_RANDOM_SEED",
    # Data
    "prepare_features",
    # Trainers
    "train_catboost",
    "train_xgboost",
    "train_lightgbm",
    # Metrics
    "evaluate_model",
    # Pipeline
    "run_evaluation",
]
