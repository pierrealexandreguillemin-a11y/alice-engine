"""Module: scripts/training/optuna_tuning.py - Optimisation Optuna.

Document ID: ALICE-MOD-OPTUNA-001
Version: 1.0.0

Ce module implémente l'optimisation bayésienne des hyperparamètres
pour CatBoost, XGBoost et LightGBM avec Optuna TPESampler.

Algorithme:
- TPE (Tree-structured Parzen Estimator) - Bergstra et al. 2011
- Direction: maximize AUC-ROC
- Early stopping: 50 rounds sans amélioration
- Seed fixe: 42 (reproductibilité ISO 42001)

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Traçabilité, Reproductibilité)
- ISO/IEC 5055:2021 - Code Quality (<300 lignes, responsabilité unique)
- ISO/IEC 29119:2022 - Software Testing (validation croisée)

Dependencies:
- optuna>=4.6.0 (TPESampler, Study, Trial)
- catboost>=1.2.7 (CatBoostClassifier)
- xgboost>=2.1.0 (XGBClassifier)
- lightgbm>=4.3.0 (LGBMClassifier)

See Also
--------
- config/hyperparameters.yaml - Configuration search space
- tests/test_training_optuna.py - Tests unitaires (12 tests)

References
----------
- Bergstra, J. et al. (2011). Algorithms for Hyper-Parameter Optimization.
  https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf
- Optuna Documentation: https://optuna.readthedocs.io/

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    config: dict[str, Any],
    model_name: str = "catboost",
) -> dict[str, Any]:
    """Optimise les hyperparamètres avec Optuna TPESampler.

    Args:
    ----
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        X_valid: Features de validation
        y_valid: Target de validation
        cat_features: Liste des colonnes catégorielles
        config: Configuration avec search space Optuna
        model_name: Modèle à optimiser ('catboost', 'xgboost', 'lightgbm')

    Returns:
    -------
        Meilleurs hyperparamètres trouvés

    Raises:
    ------
        ValueError: Si model_name n'est pas supporté

    ISO 42001: Traçabilité complète du processus d'optimisation.
    """
    import optuna

    # Désactiver les logs verbeux d'Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    optuna_config = config.get("optuna", {})
    n_trials = optuna_config.get("n_trials", 50)
    timeout = optuna_config.get("timeout", 1800)  # 30 min par défaut

    logger.info(f"[Optuna] Starting {model_name} optimization...")
    logger.info(f"[Optuna] n_trials={n_trials}, timeout={timeout}s")

    # Créer l'objectif selon le modèle
    if model_name == "catboost":
        objective = _create_catboost_objective(
            X_train, y_train, X_valid, y_valid, cat_features, optuna_config
        )
    elif model_name == "xgboost":
        objective = _create_xgboost_objective(X_train, y_train, X_valid, y_valid, optuna_config)
    elif model_name == "lightgbm":
        objective = _create_lightgbm_objective(
            X_train, y_train, X_valid, y_valid, cat_features, optuna_config
        )
    else:
        raise ValueError(f"Modèle non supporté: {model_name}")

    # Lancer l'optimisation (ISO 42001: seed fixe pour reproductibilité)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"alice_{model_name}_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    logger.info(f"[Optuna] Best AUC: {study.best_value:.4f}")
    logger.info(f"[Optuna] Best params: {study.best_params}")

    return study.best_params


def _create_catboost_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    optuna_config: dict[str, Any],
) -> Any:
    """Crée la fonction objectif pour CatBoost.

    Search Space (ISO 42001 - documenté):
    - iterations: Nombre d'arbres [500, 1000, 1500]
    - learning_rate: Taux d'apprentissage [0.01, 0.03, 0.05, 0.1]
    - depth: Profondeur max des arbres [4, 6, 8, 10]
    - l2_leaf_reg: Régularisation L2 [1, 3, 5, 7]
    - min_data_in_leaf: Samples min par feuille [10, 20, 50, 100]
    """
    search_space = optuna_config.get("catboost_search_space", {})

    def objective(trial: Any) -> float:
        params = {
            "iterations": trial.suggest_categorical(
                "iterations", search_space.get("iterations", [500, 1000])
            ),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", search_space.get("learning_rate", [0.01, 0.03, 0.05])
            ),
            "depth": trial.suggest_categorical("depth", search_space.get("depth", [4, 6, 8])),
            "l2_leaf_reg": trial.suggest_categorical(
                "l2_leaf_reg", search_space.get("l2_leaf_reg", [1, 3, 5])
            ),
            "min_data_in_leaf": trial.suggest_categorical(
                "min_data_in_leaf", search_space.get("min_data_in_leaf", [10, 20, 50])
            ),
            "cat_features": cat_features,
            "early_stopping_rounds": 50,
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": 0,
        }

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

        return model.get_best_score()["validation"]["AUC"]

    return objective


def _create_xgboost_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    optuna_config: dict[str, Any],
) -> Any:
    """Crée la fonction objectif pour XGBoost.

    Search Space (ISO 42001 - documenté):
    - n_estimators: Nombre d'arbres [500, 1000, 1500]
    - learning_rate: Taux d'apprentissage [0.01, 0.03, 0.05, 0.1]
    - max_depth: Profondeur max [4, 6, 8, 10]
    - reg_lambda: Régularisation L2 [0.1, 1.0, 5.0, 10.0]
    - colsample_bytree: Fraction features par arbre [0.7, 0.8, 0.9, 1.0]
    - subsample: Fraction samples par arbre [0.8, 0.9, 1.0]
    """
    search_space = optuna_config.get("xgboost_search_space", {})

    def objective(trial: Any) -> float:
        params = {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", search_space.get("n_estimators", [500, 1000])
            ),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", search_space.get("learning_rate", [0.01, 0.03, 0.05])
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", search_space.get("max_depth", [4, 6, 8])
            ),
            "reg_lambda": trial.suggest_categorical(
                "reg_lambda", search_space.get("reg_lambda", [0.1, 1.0, 5.0])
            ),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", search_space.get("colsample_bytree", [0.7, 0.8, 0.9, 1.0])
            ),
            "subsample": trial.suggest_categorical(
                "subsample", search_space.get("subsample", [0.8, 0.9, 1.0])
            ),
            "early_stopping_rounds": 50,
            "eval_metric": "auc",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        return model.best_score

    return objective


def _create_lightgbm_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    optuna_config: dict[str, Any],
) -> Any:
    """Crée la fonction objectif pour LightGBM.

    Search Space (ISO 42001 - documenté):
    - n_estimators: Nombre d'arbres [500, 1000, 1500]
    - learning_rate: Taux d'apprentissage [0.01, 0.03, 0.05, 0.1]
    - num_leaves: Feuilles max par arbre [31, 63, 127, 255]
    - reg_lambda: Régularisation L2 [0.1, 1.0, 5.0, 10.0]
    - feature_fraction: Fraction features par itération [0.7, 0.8, 0.9]
    - bagging_fraction: Fraction samples par itération [0.8, 0.9, 1.0]
    - min_child_samples: Samples min par feuille [10, 20, 50, 100]
    """
    search_space = optuna_config.get("lightgbm_search_space", {})
    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]

    def objective(trial: Any) -> float:
        params = {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", search_space.get("n_estimators", [500, 1000])
            ),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", search_space.get("learning_rate", [0.01, 0.03, 0.05])
            ),
            "num_leaves": trial.suggest_categorical(
                "num_leaves", search_space.get("num_leaves", [31, 63, 127])
            ),
            "reg_lambda": trial.suggest_categorical(
                "reg_lambda", search_space.get("reg_lambda", [0.1, 1.0, 5.0])
            ),
            "feature_fraction": trial.suggest_categorical(
                "feature_fraction", search_space.get("feature_fraction", [0.7, 0.8, 0.9])
            ),
            "bagging_fraction": trial.suggest_categorical(
                "bagging_fraction", search_space.get("bagging_fraction", [0.8, 0.9, 1.0])
            ),
            "bagging_freq": 1,  # Required when bagging_fraction < 1.0
            "min_child_samples": trial.suggest_categorical(
                "min_child_samples", search_space.get("min_child_samples", [10, 20, 50])
            ),
            "categorical_feature": cat_indices,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ],
        )

        return model.best_score_["valid_0"]["auc"]

    return objective


def optimize_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Optimise les 3 modèles séquentiellement.

    Args:
    ----
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        X_valid: Features de validation
        y_valid: Target de validation
        cat_features: Liste des colonnes catégorielles
        config: Configuration avec search space Optuna

    Returns:
    -------
        Dict avec les meilleurs params pour chaque modèle:
        {"catboost": {...}, "xgboost": {...}, "lightgbm": {...}}

    ISO 42001: Traçabilité multi-modèles.
    """
    results = {}

    for model_name in ["catboost", "xgboost", "lightgbm"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizing {model_name}...")
        logger.info("=" * 60)

        best_params = optimize_hyperparameters(
            X_train, y_train, X_valid, y_valid, cat_features, config, model_name
        )
        results[model_name] = best_params

    return results
