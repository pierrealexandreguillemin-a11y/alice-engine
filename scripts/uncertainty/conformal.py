"""Conformal Prediction - ISO 24029 (Robustness).

Module de quantification d'incertitude via conformal prediction.

ISO Compliance:
- ISO/IEC 24029:2021 - Neural Network Robustness (uncertainty quantification)
- ISO/IEC 5055:2021 - Code Quality (SRP)

Conformal Prediction garantit une couverture valide:
    P(Y ∈ C(X)) ≥ 1 - α

Usage:
    from scripts.uncertainty import ConformalPredictor, UncertaintyConfig

    cp = ConformalPredictor(UncertaintyConfig(alpha=0.10))
    cp.fit(model, X_calib, y_calib)
    result = cp.predict(X_test)

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from scripts.uncertainty.uncertainty_types import (
    PredictionInterval,
    UncertaintyConfig,
    UncertaintyMethod,
    UncertaintyMetrics,
    UncertaintyResult,
)

logger = logging.getLogger(__name__)

# Minimum samples for reliable quantile estimation (ISO 24029)
MIN_CALIBRATION_SAMPLES = 30


class ConformalPredictor:
    """Prédicteur conformal pour classification binaire (ISO 24029).

    Implémente la méthode LAC (Least Ambiguous set-valued Classifier)
    pour produire des ensembles de prédiction avec garantie de couverture.

    Example:
        config = UncertaintyConfig(alpha=0.10)  # 90% coverage
        cp = ConformalPredictor(config)

        # Calibrer sur données de validation
        cp.fit(model, X_calib, y_calib)

        # Prédire avec intervalles
        result = cp.predict(X_test)
        for interval in result.intervals:
            print(f"P={interval.point_estimate:.2f} [{interval.lower:.2f}, {interval.upper:.2f}]")
    """

    def __init__(self, config: UncertaintyConfig | None = None) -> None:
        """Initialise le prédicteur conformal."""
        self.config = config or UncertaintyConfig()
        self._model: Any = None
        self._calibration_scores: np.ndarray | None = None
        self._quantile: float | None = None

    def fit(
        self,
        model: Any,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
    ) -> None:
        """Calibre le prédicteur conformal (split conformal).

        Algorithme:
        1. Calcule les probabilités sur les données de calibration
        2. Score de non-conformité = 1 - p(y_true) pour chaque exemple
        3. Calcule le quantile (1-α)(n+1)/n des scores

        Args:
            model: Modèle de base avec predict_proba
            X: Features de calibration
            y: Labels de calibration (0/1)
        """
        self._model = model
        X_arr = np.asarray(X) if hasattr(X, "values") else X
        y_arr = np.asarray(y)

        # Validation taille minimale (ISO 24029 - robustesse statistique)
        if len(y_arr) < MIN_CALIBRATION_SAMPLES:
            logger.warning(
                f"Calibration set size ({len(y_arr)}) below recommended minimum "
                f"({MIN_CALIBRATION_SAMPLES}). Quantile estimation may be unreliable."
            )

        # Probabilités complètes (classe 0 et classe 1)
        # AutoGluon retourne DataFrame, sklearn retourne ndarray
        proba_raw = model.predict_proba(X)
        proba_all = proba_raw.values if hasattr(proba_raw, "values") else proba_raw

        # Score de non-conformité: 1 - p(y_true)
        # Pour y=1: score = 1 - proba[:,1]
        # Pour y=0: score = 1 - proba[:,0] = proba[:,1]
        self._calibration_scores = np.where(
            y_arr == 1,
            1 - proba_all[:, 1],  # y=1: 1 - P(Y=1)
            proba_all[:, 1],  # y=0: 1 - P(Y=0) = P(Y=1)
        )

        # Quantile pour couverture 1 - alpha (formule standard)
        n = len(self._calibration_scores)
        q_level = min(np.ceil((n + 1) * (1 - self.config.alpha)) / n, 1.0)
        self._quantile = float(np.quantile(self._calibration_scores, q_level))

        logger.info(
            f"Conformal calibration: n={n}, alpha={self.config.alpha:.2f}, "
            f"quantile_level={q_level:.4f}, threshold={self._quantile:.4f}"
        )

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        y_true: np.ndarray | None = None,
    ) -> UncertaintyResult:
        """Prédit avec ensembles de prédiction conformaux.

        Algorithme:
        - Classe y est dans l'ensemble si: 1 - p(y) <= q_hat
        - Équivalent: p(y) >= 1 - q_hat

        Args:
            X: Features de test
            y_true: Labels réels optionnels (pour métriques de couverture)

        Returns:
            UncertaintyResult avec intervalles et métriques
        """
        if self._model is None or self._quantile is None:
            raise RuntimeError("Conformal predictor not fitted. Call fit() first.")

        # Probabilités complètes (AutoGluon retourne DataFrame)
        proba_raw = self._model.predict_proba(X)
        proba_all = proba_raw.values if hasattr(proba_raw, "values") else proba_raw
        threshold = 1 - self._quantile  # Seuil d'inclusion

        intervals = []
        for i in range(len(proba_all)):
            p0 = proba_all[i, 0]  # P(Y=0)
            p1 = proba_all[i, 1]  # P(Y=1)

            # Ensemble de prédiction: classes où p(y) >= 1 - q_hat
            pred_set = []
            if p0 >= threshold:
                pred_set.append(0)
            if p1 >= threshold:
                pred_set.append(1)

            # Intervalle basé sur l'ensemble de prédiction
            # Si les deux classes sont dans l'ensemble: [0, 1]
            # Si une seule classe: intervalle resserré
            if len(pred_set) == 2:
                lower, upper = 0.0, 1.0
            elif pred_set == [1]:
                lower, upper = threshold, 1.0
            elif pred_set == [0]:
                lower, upper = 0.0, 1 - threshold
            else:  # Ensemble vide (rare)
                lower, upper = p1, p1

            intervals.append(
                PredictionInterval(
                    point_estimate=float(p1),
                    lower=lower,
                    upper=upper,
                    confidence=1 - self.config.alpha,
                    in_prediction_set=sorted(pred_set),
                )
            )

        # Métriques
        widths = [i.interval_width() for i in intervals]
        empty_count = sum(1 for i in intervals if len(i.in_prediction_set) == 0)
        singleton_count = sum(1 for i in intervals if len(i.in_prediction_set) == 1)

        coverage = 0.0
        if y_true is not None:
            y_arr = np.asarray(y_true)
            coverage = np.mean([y_arr[i] in intervals[i].in_prediction_set for i in range(len(y_arr))])

        metrics = UncertaintyMetrics(
            mean_interval_width=float(np.mean(widths)),
            coverage=float(coverage),
            efficiency=1 - (len(intervals) - singleton_count) / len(intervals) if intervals else 0,
            empty_set_rate=empty_count / len(intervals) if intervals else 0,
            singleton_rate=singleton_count / len(intervals) if intervals else 0,
        )

        logger.info(
            f"Conformal prediction: n={len(intervals)}, "
            f"coverage={metrics.coverage:.2%}, width={metrics.mean_interval_width:.3f}"
        )

        return UncertaintyResult(
            intervals=intervals,
            metrics=metrics,
            method=UncertaintyMethod.CONFORMAL,
            calibration_scores=self._calibration_scores.tolist() if self._calibration_scores is not None else [],
        )


def quantify_uncertainty(
    model: Any,
    X_calib: np.ndarray | pd.DataFrame,
    y_calib: np.ndarray,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | None = None,
    alpha: float = 0.10,
) -> UncertaintyResult:
    """Fonction utilitaire pour quantification d'incertitude.

    Args:
        model: Modèle de base
        X_calib: Features de calibration
        y_calib: Labels de calibration
        X_test: Features de test
        y_test: Labels de test optionnels
        alpha: Niveau de significativité

    Returns:
        UncertaintyResult
    """
    config = UncertaintyConfig(alpha=alpha)
    cp = ConformalPredictor(config)
    cp.fit(model, X_calib, y_calib)
    return cp.predict(X_test, y_test)
