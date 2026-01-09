"""Configuration et hyperparamètres pour l'entraînement - ISO 5055.

Ce module contient les fonctions de gestion de la configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_hyperparameters(config_path: Path) -> dict[str, object]:
    """Charge les hyperparametres depuis un fichier YAML."""
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return get_default_hyperparameters()

    with config_path.open() as f:
        result: dict[str, object] = yaml.safe_load(f)
        return result


def get_default_hyperparameters() -> dict[str, object]:
    """Retourne les hyperparametres par defaut."""
    return {
        "global": {
            "random_seed": 42,
            "early_stopping_rounds": 50,
            "verbose": 100,
        },
        "catboost": {
            "iterations": 1000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "verbose": 100,
            "early_stopping_rounds": 50,
        },
        "xgboost": {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 6,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 1,
            "early_stopping_rounds": 50,
        },
        "lightgbm": {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
            "early_stopping_rounds": 50,
        },
    }


def get_config_value(config: dict[str, object], key: str, default: object) -> object:
    """Extrait une valeur de config avec default."""
    return config.get(key, default)
