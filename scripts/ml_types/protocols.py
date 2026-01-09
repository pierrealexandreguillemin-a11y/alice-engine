"""Protocols ML - Interfaces pour modeles.

Ce module definit les protocols (interfaces) pour les modeles ML:
- MLClassifier: Interface sklearn-compatible generique
- CatBoostModel: Protocol specifique CatBoost
- XGBoostModel: Protocol specifique XGBoost
- LightGBMModel: Protocol specifique LightGBM

Conformite:
- ISO/IEC 5055 (Code Quality)
- PEP 544 (Protocols)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray


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
