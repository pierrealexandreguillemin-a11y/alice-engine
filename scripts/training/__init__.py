"""Package Training - ISO 5055.

Ce package contient les modules pour l'entraînement des modèles:
- config.py: Chargement des hyperparamètres
- features.py: Préparation des features
- metrics.py: Calcul des métriques
- trainers.py: Fonctions d'entraînement
- parallel.py: Exécution parallèle
- mlflow_tracking.py: Intégration MLflow
"""

from scripts.training.config import (
    get_config_value,
    get_default_hyperparameters,
    load_hyperparameters,
)
from scripts.training.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_features,
)
from scripts.training.metrics import compute_all_metrics
from scripts.training.mlflow_tracking import log_to_mlflow, setup_mlflow
from scripts.training.parallel import train_all_models_parallel
from scripts.training.trainers import train_catboost, train_lightgbm, train_xgboost

__all__ = [
    # Config
    "load_hyperparameters",
    "get_default_hyperparameters",
    "get_config_value",
    # Features
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "prepare_features",
    # Metrics
    "compute_all_metrics",
    # Trainers
    "train_catboost",
    "train_xgboost",
    "train_lightgbm",
    # Parallel
    "train_all_models_parallel",
    # MLflow
    "setup_mlflow",
    "log_to_mlflow",
]
