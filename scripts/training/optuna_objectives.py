"""Objectifs Optuna pour les modeles - ISO 42001.

Ce module contient les fonctions objectif pour Optuna:
- CatBoost, XGBoost, LightGBM

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Tracabilite)
- ISO/IEC 5055:2021 - Code Quality (<200 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

if TYPE_CHECKING:
    import pandas as pd


def create_catboost_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    optuna_config: dict[str, Any],
) -> Any:
    """Cree la fonction objectif pour CatBoost.

    Search Space (ISO 42001 - documente):
    - iterations: Nombre d'arbres [500, 1000, 1500]
    - learning_rate: Taux d'apprentissage [0.01, 0.03, 0.05, 0.1]
    - depth: Profondeur max des arbres [4, 6, 8, 10]
    - l2_leaf_reg: Regularisation L2 [1, 3, 5, 7]
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


def create_xgboost_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    optuna_config: dict[str, Any],
) -> Any:
    """Cree la fonction objectif pour XGBoost.

    Search Space (ISO 42001 - documente):
    - n_estimators: Nombre d'arbres [500, 1000, 1500]
    - learning_rate: Taux d'apprentissage [0.01, 0.03, 0.05, 0.1]
    - max_depth: Profondeur max [4, 6, 8, 10]
    - reg_lambda: Regularisation L2 [0.1, 1.0, 5.0, 10.0]
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


def create_lightgbm_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    optuna_config: dict[str, Any],
) -> Any:
    """Cree la fonction objectif pour LightGBM.

    Search Space (ISO 42001 - documente):
    - n_estimators: Nombre d'arbres [500, 1000, 1500]
    - learning_rate: Taux d'apprentissage [0.01, 0.03, 0.05, 0.1]
    - num_leaves: Feuilles max par arbre [31, 63, 127, 255]
    - reg_lambda: Regularisation L2 [0.1, 1.0, 5.0, 10.0]
    - feature_fraction: Fraction features par iteration [0.7, 0.8, 0.9]
    - bagging_fraction: Fraction samples par iteration [0.8, 0.9, 1.0]
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
            "bagging_freq": 1,
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


# Aliases pour compatibilite (prefixe _ historique)
_create_catboost_objective = create_catboost_objective
_create_xgboost_objective = create_xgboost_objective
_create_lightgbm_objective = create_lightgbm_objective
