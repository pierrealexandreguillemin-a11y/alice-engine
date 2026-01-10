"""Module: scripts/autogluon/predictor_wrapper.py - sklearn-compatible Wrapper.

Document ID: ALICE-MOD-AUTOGLUON-WRAPPER-001
Version: 1.0.0

Wrapper sklearn-compatible pour AutoGluon TabularPredictor.
Permet l'integration avec les pipelines existants.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (interoperabilite)
- ISO/IEC 5055:2021 - Code Quality (<200 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class AutoGluonWrapper:
    """Wrapper sklearn-compatible pour AutoGluon TabularPredictor.

    Cette classe permet d'utiliser AutoGluon avec les interfaces
    sklearn standard (fit, predict, predict_proba).

    ISO 42001: Interoperabilite avec ecosysteme ML.

    Attributes
    ----------
        predictor: TabularPredictor AutoGluon sous-jacent
        label: Nom de la colonne cible
        feature_names: Noms des features
    """

    def __init__(
        self,
        predictor: TabularPredictor | None = None,
        label: str = "target",
    ) -> None:
        """Initialise le wrapper.

        Args:
        ----
            predictor: TabularPredictor existant (optionnel)
            label: Nom de la colonne cible
        """
        self.predictor = predictor
        self.label = label
        self.feature_names: list[str] = []
        self._is_fitted = predictor is not None

    def fit(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
        y: NDArray[np.int64] | pd.Series,
        **fit_params: Any,
    ) -> AutoGluonWrapper:
        """Entraine le modele AutoGluon.

        Args:
        ----
            X: Features d'entrainement
            y: Labels cibles
            **fit_params: Parametres additionnels pour TabularPredictor.fit()

        Returns:
        -------
            self (pour chaining)
        """
        from autogluon.tabular import TabularPredictor

        # Convertir en DataFrame si necessaire
        if isinstance(X, np.ndarray):
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names)
        else:
            self.feature_names = list(X.columns)

        # Creer DataFrame avec label
        train_data = X.copy()
        train_data[self.label] = y

        # Creer et entrainer le predictor
        self.predictor = TabularPredictor(
            label=self.label,
            **fit_params.get("predictor_kwargs", {}),
        )

        self.predictor.fit(
            train_data=train_data,
            **{k: v for k, v in fit_params.items() if k != "predictor_kwargs"},
        )

        self._is_fitted = True
        return self

    def predict(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
    ) -> NDArray[np.int64]:
        """Predit les labels.

        Args:
        ----
            X: Features

        Returns:
        -------
            Predictions

        Raises:
        ------
            ValueError: Si le modele n'est pas entraine
        """
        self._check_is_fitted()
        X_df = self._prepare_input(X)
        return self.predictor.predict(X_df).values

    def predict_proba(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
    ) -> NDArray[np.float64]:
        """Predit les probabilites.

        Args:
        ----
            X: Features

        Returns:
        -------
            Probabilites pour chaque classe

        Raises:
        ------
            ValueError: Si le modele n'est pas entraine
        """
        self._check_is_fitted()
        X_df = self._prepare_input(X)
        proba = self.predictor.predict_proba(X_df)

        # Retourner au format sklearn (n_samples, n_classes)
        if isinstance(proba, pd.DataFrame):
            return proba.values
        return proba

    def score(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
        y: NDArray[np.int64] | pd.Series,
    ) -> float:
        """Calcule le score du modele.

        Args:
        ----
            X: Features
            y: Labels vrais

        Returns:
        -------
            Score (metrique par defaut du predictor)
        """
        self._check_is_fitted()
        X_df = self._prepare_input(X)
        test_data = X_df.copy()
        test_data[self.label] = y

        metrics = self.predictor.evaluate(test_data)
        # Retourner la metrique principale
        if isinstance(metrics, dict):
            return list(metrics.values())[0]
        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Retourne l'importance des features.

        Returns
        -------
            DataFrame avec importance par feature
        """
        self._check_is_fitted()
        return self.predictor.feature_importance()

    def save(self, path: str | Path) -> None:
        """Sauvegarde le modele.

        Args:
        ----
            path: Chemin de sauvegarde
        """
        self._check_is_fitted()
        self.predictor.save(str(path))
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> AutoGluonWrapper:
        """Charge un modele sauvegarde.

        Args:
        ----
            path: Chemin du modele

        Returns:
        -------
            AutoGluonWrapper avec le predictor charge
        """
        from autogluon.tabular import TabularPredictor

        predictor = TabularPredictor.load(str(path))
        wrapper = cls(predictor=predictor, label=predictor.label)
        wrapper.feature_names = list(predictor.feature_metadata.get_features())
        return wrapper

    def _check_is_fitted(self) -> None:
        """Verifie que le modele est entraine."""
        if not self._is_fitted or self.predictor is None:
            msg = "Model not fitted. Call fit() first."
            raise ValueError(msg)

    def _prepare_input(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare les donnees d'entree.

        Args:
        ----
            X: Features (array ou DataFrame)

        Returns:
        -------
            DataFrame avec les bons noms de colonnes
        """
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=self.feature_names)
        return X
