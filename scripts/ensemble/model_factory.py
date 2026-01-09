"""Factory de modeles pour Ensemble - ISO 5055.

Ce module contient les fonctions de creation de modeles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.ml_types import MLClassifier


def create_catboost_model(
    params: dict[str, object],
    cat_indices: list[int],
) -> MLClassifier:
    """Cree une instance CatBoost."""
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=params.get("iterations", 1000),
        learning_rate=params.get("learning_rate", 0.03),
        depth=params.get("depth", 6),
        l2_leaf_reg=params.get("l2_leaf_reg", 3),
        cat_features=cat_indices,
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        eval_metric="AUC",
        random_seed=params.get("random_seed", 42),
        verbose=0,
        thread_count=-1,
    )


def create_xgboost_model(params: dict[str, object]) -> MLClassifier:
    """Cree une instance XGBoost."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.03),
        max_depth=params.get("max_depth", 6),
        reg_lambda=params.get("reg_lambda", 1.0),
        tree_method="hist",
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        eval_metric="auc",
        random_state=params.get("random_state", 42),
        verbosity=0,
        n_jobs=-1,
    )


def create_lightgbm_model(
    params: dict[str, object],
    cat_indices: list[int],
) -> MLClassifier:
    """Cree une instance LightGBM."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.03),
        num_leaves=params.get("num_leaves", 63),
        reg_lambda=params.get("reg_lambda", 1.0),
        categorical_feature=cat_indices,
        random_state=params.get("random_state", 42),
        verbose=-1,
        n_jobs=-1,
    )


def create_model_by_name(
    name: str,
    params: dict[str, object],
    cat_indices: list[int],
) -> MLClassifier:
    """Cree un modele par son nom."""
    if name == "CatBoost":
        return create_catboost_model(params, cat_indices)
    if name == "XGBoost":
        return create_xgboost_model(params)
    if name == "LightGBM":
        return create_lightgbm_model(params, cat_indices)
    raise ValueError(f"Unknown model: {name}")
