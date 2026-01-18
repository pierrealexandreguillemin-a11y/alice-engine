"""Confidence Calibrator - ISO 24029 (Robustness).

Module de calibration des probabilités ML via Platt scaling ou isotonic regression.

ISO Compliance:
- ISO/IEC 24029:2021 - Neural Network Robustness (calibrated outputs)
- ISO/IEC 5055:2021 - Code Quality (SRP)

Usage:
    from scripts.calibration import Calibrator, CalibrationConfig

    calibrator = Calibrator(CalibrationConfig(method=CalibrationMethod.ISOTONIC))
    result = calibrator.fit(model, X_calib, y_calib)
    calibrated_proba = result.calibrator.predict_proba(X_test)[:, 1]

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

if TYPE_CHECKING:
    import pandas as pd

from scripts.calibration.calibrator_types import (
    CalibrationConfig,
    CalibrationMethod,
    CalibrationMetrics,
    CalibrationResult,
)

logger = logging.getLogger(__name__)


class AutoGluonWrapper:
    """Wrapper sklearn-compatible pour AutoGluon TabularPredictor."""

    _estimator_type = "classifier"  # Requis pour sklearn

    def __init__(self, predictor: Any, feature_names: list[str] | None = None) -> None:
        self.predictor = predictor
        self.classes_ = np.array([0, 1])
        self._feature_names = feature_names

    def fit(self, X: Any, y: Any) -> "AutoGluonWrapper":
        """No-op: AutoGluon est déjà entraîné. Capture feature names."""
        if hasattr(X, "columns"):
            self._feature_names = list(X.columns)
        return self

    def _to_dataframe(self, X: Any) -> Any:
        """Convertit en DataFrame si nécessaire (pour AutoGluon)."""
        if hasattr(X, "columns"):
            return X
        import pandas as pd
        if self._feature_names:
            return pd.DataFrame(X, columns=self._feature_names)
        return pd.DataFrame(X)

    def predict(self, X: Any) -> np.ndarray:
        """Prédit les classes."""
        X_df = self._to_dataframe(X)
        return self.predictor.predict(X_df).values

    def predict_proba(self, X: Any) -> np.ndarray:
        """Prédit les probabilités."""
        X_df = self._to_dataframe(X)
        proba = self.predictor.predict_proba(X_df)
        if hasattr(proba, "values"):
            return proba.values
        return proba


def _wrap_if_autogluon(model: Any, feature_names: list[str] | None = None) -> Any:
    """Wrap AutoGluon predictor si nécessaire."""
    # Détecter AutoGluon par le nom de la classe
    model_class = type(model).__name__
    if "TabularPredictor" in model_class or "Predictor" in model_class:
        return AutoGluonWrapper(model, feature_names)
    return model


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calcule l'Expected Calibration Error (ECE).

    ECE mesure l'écart moyen pondéré entre confiance et accuracy.
    Plus ECE est proche de 0, meilleure est la calibration.

    Args:
        y_true: Labels réels (0/1)
        y_prob: Probabilités prédites
        n_bins: Nombre de bins

    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(y_prob[in_bin])
            avg_accuracy = np.mean(y_true[in_bin])
            ece += prop_in_bin * np.abs(avg_accuracy - avg_confidence)

    return float(ece)


class Calibrator:
    """Calibrateur de probabilités ML (ISO 24029).

    Utilise sklearn CalibratedClassifierCV pour calibrer
    les probabilités via Platt scaling ou isotonic regression.

    Example:
        config = CalibrationConfig(method=CalibrationMethod.PLATT)
        calibrator = Calibrator(config)

        # Calibrer sur données de validation
        result = calibrator.fit(base_model, X_calib, y_calib)

        # Utiliser le modèle calibré
        proba = result.calibrator.predict_proba(X_test)[:, 1]
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        """Initialise le calibrateur."""
        self.config = config or CalibrationConfig()

    def fit(
        self,
        base_model: Any,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
    ) -> CalibrationResult:
        """Calibre un modèle sur les données de calibration.

        Args:
            base_model: Modèle de base avec predict_proba
            X: Features de calibration
            y: Labels de calibration

        Returns:
            CalibrationResult avec modèle calibré et métriques
        """
        # Wrapper pour AutoGluon si nécessaire
        feature_names = list(X.columns) if hasattr(X, "columns") else None
        wrapped_model = _wrap_if_autogluon(base_model, feature_names)
        X_arr = np.asarray(X) if hasattr(X, "values") else X
        y_arr = np.asarray(y)

        # Probabilités avant calibration
        proba_before = wrapped_model.predict_proba(X)[:, 1]

        # Métriques avant
        brier_before = brier_score_loss(y_arr, proba_before)
        ece_before = compute_ece(y_arr, proba_before, self.config.n_bins)

        # Calibration
        method_str = "sigmoid" if self.config.method == CalibrationMethod.PLATT else "isotonic"

        # cv=0 signifie "prefit" - modèle déjà entraîné, utiliser FrozenEstimator
        # cv>0 signifie cross-validation pour calibration
        if self.config.cv == 0:
            # FrozenEstimator remplace cv="prefit" (deprecated sklearn 1.6+)
            calibrated = CalibratedClassifierCV(
                estimator=FrozenEstimator(wrapped_model),
                method=method_str,
            )
        else:
            calibrated = CalibratedClassifierCV(
                estimator=wrapped_model,
                method=method_str,
                cv=self.config.cv,
            )
        calibrated.fit(X_arr, y_arr)

        # Probabilités après calibration
        proba_after = calibrated.predict_proba(X_arr)[:, 1]

        # Métriques après
        brier_after = brier_score_loss(y_arr, proba_after)
        ece_after = compute_ece(y_arr, proba_after, self.config.n_bins)

        # Amélioration
        improvement = ((brier_before - brier_after) / brier_before) * 100 if brier_before > 0 else 0

        # Courbe de calibration
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_arr, proba_after, n_bins=self.config.n_bins
        )

        metrics = CalibrationMetrics(
            brier_before=brier_before,
            brier_after=brier_after,
            ece_before=ece_before,
            ece_after=ece_after,
            improvement_pct=improvement,
        )

        logger.info(
            f"Calibration ({method_str}): Brier {brier_before:.4f} -> {brier_after:.4f} "
            f"({improvement:+.1f}%), ECE {ece_before:.4f} -> {ece_after:.4f}"
        )

        return CalibrationResult(
            calibrator=calibrated,
            method=self.config.method,
            metrics=metrics,
            calibration_curve={
                "fraction_positives": fraction_of_positives.tolist(),
                "mean_predicted": mean_predicted_value.tolist(),
            },
        )


def calibrate_model(
    base_model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    method: CalibrationMethod = CalibrationMethod.ISOTONIC,
    cv: int = 5,
) -> CalibrationResult:
    """Fonction utilitaire pour calibration rapide.

    Args:
        base_model: Modèle de base
        X: Features de calibration
        y: Labels de calibration
        method: Méthode de calibration
        cv: Nombre de folds CV (0 = prefit, modèle déjà entraîné)

    Returns:
        CalibrationResult
    """
    config = CalibrationConfig(method=method, cv=cv)
    calibrator = Calibrator(config)
    return calibrator.fit(base_model, X, y)
