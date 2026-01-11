"""Explicabilite SHAP - ISO 42001.

Ce module fournit l'explicabilite des modeles via SHAP et permutation importance.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management (Explicabilite)
- ISO/IEC 5055 - Code Quality (<300 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap_importance(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    *,
    max_samples: int = 100,
) -> dict[str, float]:
    """Calcule l'importance SHAP des features.

    Args:
    ----
        model: Modele entraine (CatBoost, XGBoost, LightGBM, sklearn)
        X: Donnees pour calcul SHAP (background dataset)
        feature_names: Noms des features
        max_samples: Nombre max d'echantillons pour calcul (performance)

    Returns:
    -------
        Dict feature_name -> importance SHAP moyenne absolue (normalise)

    ISO 42001: Explicabilite via SHAP values.
    """
    try:
        # Limiter les echantillons pour performance
        if hasattr(X, "values"):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        if len(X_arr) > max_samples:
            indices = np.random.choice(len(X_arr), max_samples, replace=False)
            X_sample = X_arr[indices]
        else:
            X_sample = X_arr

        # Creer l'explainer adapte au type de modele
        explainer = _create_shap_explainer(model, X_sample)
        if explainer is None:
            logger.warning("Could not create SHAP explainer for this model type")
            return {}

        # Calculer les SHAP values
        shap_values = explainer(X_sample)

        # Extraire les valeurs moyennes absolues
        if hasattr(shap_values, "values"):
            values = shap_values.values
        else:
            values = shap_values

        # Gerer multi-output (classification binaire)
        if len(values.shape) == 3:
            values = values[:, :, 1]  # Prendre la classe positive

        mean_abs_shap = np.abs(values).mean(axis=0)

        # Construire le dictionnaire normalise
        importance = _build_normalized_importance(mean_abs_shap, feature_names)

        logger.info(f"  SHAP importance computed for {len(importance)} features")
        return importance

    except ImportError:
        logger.warning("SHAP not installed, skipping SHAP explainability")
        return {}
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return {}


def _create_shap_explainer(model: Any, X_background: np.ndarray) -> Any:
    """Cree l'explainer SHAP adapte au modele."""
    import shap

    model_type = type(model).__name__

    # Tree-based models (dispatch table pattern)
    tree_models = {
        "CatBoostClassifier",
        "CatBoostRegressor",
        "XGBClassifier",
        "XGBRegressor",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
    }

    if model_type in tree_models or "LGBM" in model_type or "LightGBM" in model_type:
        return shap.TreeExplainer(model)

    # Fallback: KernelExplainer (plus lent mais universel)
    predict_fn = getattr(model, "predict_proba", None) or getattr(model, "predict", None)
    if predict_fn:
        return shap.KernelExplainer(predict_fn, X_background[:50])

    return None


def _build_normalized_importance(
    values: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Construit le dictionnaire d'importance normalise."""
    if len(values) != len(feature_names):
        logger.warning(f"Feature count mismatch: {len(values)} vs {len(feature_names)}")
        feature_names = [f"feature_{i}" for i in range(len(values))]

    importance = {name: float(values[i]) for i, name in enumerate(feature_names)}

    # Normaliser pour sommer a 1
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}

    # Trier par importance decroissante
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def compute_permutation_importance(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict[str, float]:
    """Calcule l'importance par permutation.

    Args:
    ----
        model: Modele entraine
        X: Features de validation
        y: Labels de validation
        feature_names: Noms des features
        n_repeats: Nombre de permutations par feature
        random_state: Seed pour reproductibilite

    Returns:
    -------
        Dict feature_name -> importance par permutation (normalise)

    ISO 42001: Explicabilite model-agnostic.
    """
    try:
        from sklearn.inspection import permutation_importance

        if hasattr(X, "values"):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        result = permutation_importance(
            model,
            X_arr,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )

        importance = _build_normalized_importance(result.importances_mean, feature_names)

        logger.info(f"  Permutation importance computed for {len(importance)} features")
        return importance

    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")
        return {}


def get_top_features(
    importance: dict[str, float],
    top_k: int = 10,
) -> dict[str, float]:
    """Retourne les top K features les plus importantes.

    Args:
    ----
        importance: Dict feature -> importance
        top_k: Nombre de features a retourner

    Returns:
    -------
        Dict des top K features
    """
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_features[:top_k])


def explain_prediction(
    model: Any,
    X_instance: np.ndarray,
    feature_names: list[str],
    X_background: np.ndarray | None = None,
) -> dict[str, float]:
    """Explique une prediction individuelle via SHAP.

    Args:
    ----
        model: Modele entraine
        X_instance: Instance a expliquer (1D array)
        feature_names: Noms des features
        X_background: Dataset de reference (optionnel)

    Returns:
    -------
        Dict feature_name -> contribution SHAP a la prediction
    """
    try:
        # Assurer que X_instance est 2D
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)

        explainer = _create_shap_explainer(model, X_background or X_instance)
        if explainer is None:
            return {}

        shap_values = explainer(X_instance)

        if hasattr(shap_values, "values"):
            values = shap_values.values[0]
        else:
            values = shap_values[0]

        # Gerer multi-output
        if len(values.shape) > 1:
            values = values[:, 1]

        return {name: float(values[i]) for i, name in enumerate(feature_names)}

    except Exception as e:
        logger.warning(f"Prediction explanation failed: {e}")
        return {}
