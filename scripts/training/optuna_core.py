"""Coeur de l'optimisation Optuna - ISO 42001.

Ce module contient les fonctions principales d'optimisation:
- optimize_hyperparameters: Optimisation d'un modele
- optimize_all_models: Optimisation des 3 modeles

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Tracabilite, Reproductibilite)
- ISO/IEC 5055:2021 - Code Quality (<120 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from scripts.training.optuna_objectives import (
    create_catboost_objective,
    create_lightgbm_objective,
    create_xgboost_objective,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Dispatch table pour les objectifs (ISO 5055 - pattern)
_OBJECTIVE_FACTORY = {
    "catboost": create_catboost_objective,
    "xgboost": create_xgboost_objective,
    "lightgbm": create_lightgbm_objective,
}


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    config: dict[str, Any],
    model_name: str = "catboost",
) -> dict[str, Any]:
    """Optimise les hyperparametres avec Optuna TPESampler.

    Args:
    ----
        X_train: Features d'entrainement
        y_train: Target d'entrainement
        X_valid: Features de validation
        y_valid: Target de validation
        cat_features: Liste des colonnes categorielles
        config: Configuration avec search space Optuna
        model_name: Modele a optimiser ('catboost', 'xgboost', 'lightgbm')

    Returns:
    -------
        Meilleurs hyperparametres trouves

    Raises:
    ------
        ValueError: Si model_name n'est pas supporte

    ISO 42001: Tracabilite complete du processus d'optimisation.
    """
    import optuna

    # Desactiver les logs verbeux d'Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    optuna_config = config.get("optuna", {})
    n_trials = optuna_config.get("n_trials", 50)
    timeout = optuna_config.get("timeout", 1800)

    logger.info(f"[Optuna] Starting {model_name} optimization...")
    logger.info(f"[Optuna] n_trials={n_trials}, timeout={timeout}s")

    # Creer l'objectif via dispatch table
    factory = _OBJECTIVE_FACTORY.get(model_name)
    if factory is None:
        raise ValueError(f"Modele non supporte: {model_name}")

    if model_name == "xgboost":
        objective = factory(X_train, y_train, X_valid, y_valid, optuna_config)
    else:
        objective = factory(X_train, y_train, X_valid, y_valid, cat_features, optuna_config)

    # Lancer l'optimisation (ISO 42001: seed fixe pour reproductibilite)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"alice_{model_name}_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    logger.info(f"[Optuna] Best AUC: {study.best_value:.4f}")
    logger.info(f"[Optuna] Best params: {study.best_params}")

    return study.best_params


def optimize_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Optimise les 3 modeles sequentiellement.

    Args:
    ----
        X_train: Features d'entrainement
        y_train: Target d'entrainement
        X_valid: Features de validation
        y_valid: Target de validation
        cat_features: Liste des colonnes categorielles
        config: Configuration avec search space Optuna

    Returns:
    -------
        Dict avec les meilleurs params pour chaque modele:
        {"catboost": {...}, "xgboost": {...}, "lightgbm": {...}}

    ISO 42001: Tracabilite multi-modeles.
    """
    results = {}

    for model_name in ["catboost", "xgboost", "lightgbm"]:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Optimizing {model_name}...")
        logger.info("=" * 60)

        best_params = optimize_hyperparameters(
            X_train, y_train, X_valid, y_valid, cat_features, config, model_name
        )
        results[model_name] = best_params

    return results
