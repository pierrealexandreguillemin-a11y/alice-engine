"""AIMMS Postprocessor - ISO/IEC 42001:2023 AI Management System.

Orchestration post-training: calibration → uncertainty → alerting.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Clause 8.2, 9.1, 10.2)
- ISO/IEC 24029:2021 - Neural Network Robustness (calibration, uncertainty)
- ISO/IEC 23894:2023 - AI Risk Management (alerting)
- ISO/IEC 5055:2021 - Code Quality (SRP)

Usage:
    from scripts.aimms import AIMSPostprocessor, AIMSConfig

    postprocessor = AIMSPostprocessor(AIMSConfig())
    result = postprocessor.run(model, X_calib, y_calib, X_test, y_test)

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np  # noqa: TC002 - Used at runtime

if TYPE_CHECKING:
    import pandas as pd

from scripts.aimms.aimms_types import (
    AIMSConfig,
    AIMSResult,
    AlertingSummary,
    CalibrationSummary,
    LifecyclePhase,
    UncertaintySummary,
)
from scripts.alerts.alert_types import AlertConfig, AlertSeverity
from scripts.calibration.calibrator import Calibrator
from scripts.calibration.calibrator_types import CalibrationConfig, CalibrationMethod
from scripts.uncertainty.conformal import ConformalPredictor
from scripts.uncertainty.uncertainty_types import UncertaintyConfig

logger = logging.getLogger(__name__)


class AIMSPostprocessor:
    """Orchestrateur AIMMS post-training (ISO 42001).

    Coordonne les 3 composants post-training:
    1. Calibration (ISO 24029) - probabilités calibrées
    2. Uncertainty (ISO 24029) - intervalles de confiance
    3. Alerting (ISO 23894) - configuration monitoring drift

    Example:
        config = AIMSConfig(enable_calibration=True)
        postprocessor = AIMSPostprocessor(config)
        result = postprocessor.run(model, X_calib, y_calib, X_test, y_test)
    """

    def __init__(self, config: AIMSConfig | None = None) -> None:
        """Initialise le postprocessor."""
        self.config = config or AIMSConfig()
        self._calibrator: Calibrator | None = None
        self._conformal: ConformalPredictor | None = None

    def run(
        self,
        model: Any,
        X_calib: np.ndarray | pd.DataFrame,
        y_calib: np.ndarray,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | None = None,
        model_version: str = "unknown",
    ) -> AIMSResult:
        """Execute le pipeline AIMMS complet.

        Args:
            model: Modèle entraîné avec predict_proba
            X_calib: Features de calibration
            y_calib: Labels de calibration
            X_test: Features de test
            y_test: Labels de test (optionnel, pour coverage)
            model_version: Version du modèle

        Returns:
            AIMSResult avec tous les résumés
        """
        result = AIMSResult.create(model_version, LifecyclePhase.VALIDATION)

        # 1. Calibration (ISO 24029)
        if self.config.enable_calibration:
            result.calibration = self._run_calibration(model, X_calib, y_calib)

        # 2. Uncertainty quantification (ISO 24029)
        if self.config.enable_uncertainty:
            result.uncertainty = self._run_uncertainty(model, X_calib, y_calib, X_test, y_test)

        # 3. Alerting configuration (ISO 23894)
        if self.config.enable_alerting:
            result.alerting = self._configure_alerting()

        # 4. Recommendations (ISO 42001 Clause 10.2)
        result.recommendations = self._generate_recommendations(result)

        logger.info(f"AIMMS postprocessing complete: {len(result.recommendations)} recommendations")
        return result

    def _run_calibration(
        self,
        model: Any,
        X_calib: np.ndarray | pd.DataFrame,
        y_calib: np.ndarray,
    ) -> CalibrationSummary:
        """Execute calibration (ISO 24029 Clause 6.4)."""
        config = CalibrationConfig(
            method=CalibrationMethod.ISOTONIC,
            cv=self.config.calibration_cv,
        )
        self._calibrator = Calibrator(config)
        result = self._calibrator.fit(model, X_calib, y_calib)

        logger.info(
            f"Calibration: Brier {result.metrics.brier_before:.4f} → "
            f"{result.metrics.brier_after:.4f} ({result.metrics.improvement_pct:+.1f}%)"
        )

        return CalibrationSummary(
            method=result.method.value,
            brier_before=result.metrics.brier_before,
            brier_after=result.metrics.brier_after,
            ece_before=result.metrics.ece_before,
            ece_after=result.metrics.ece_after,
            improvement_pct=result.metrics.improvement_pct,
        )

    def _run_uncertainty(
        self,
        model: Any,
        X_calib: np.ndarray | pd.DataFrame,
        y_calib: np.ndarray,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | None,
    ) -> UncertaintySummary:
        """Execute uncertainty quantification (ISO 24029 Clause 6.5)."""
        config = UncertaintyConfig(alpha=self.config.uncertainty_alpha)
        self._conformal = ConformalPredictor(config)
        self._conformal.fit(model, X_calib, y_calib)
        result = self._conformal.predict(X_test, y_test)

        logger.info(
            f"Uncertainty: coverage={result.metrics.coverage:.2%}, "
            f"width={result.metrics.mean_interval_width:.3f}"
        )

        return UncertaintySummary(
            method=result.method.value,
            coverage=result.metrics.coverage,
            mean_interval_width=result.metrics.mean_interval_width,
            singleton_rate=result.metrics.singleton_rate,
            empty_set_rate=result.metrics.empty_set_rate,
        )

    def _configure_alerting(self) -> AlertingSummary:
        """Configure alerting drift (ISO 23894 Clause 6.1)."""
        slack_url = os.environ.get("ALICE_SLACK_WEBHOOK", "")
        config = AlertConfig(
            slack_webhook_url=slack_url,
            enable_slack=bool(slack_url),
            min_severity=AlertSeverity.WARNING,
            cooldown_minutes=60,
        )

        logger.info(f"Alerting: Slack={'configured' if slack_url else 'not configured'}")

        return AlertingSummary(
            enabled=True,
            slack_configured=bool(slack_url),
            min_severity=config.min_severity.to_string(),
            cooldown_minutes=config.cooldown_minutes,
        )

    def _generate_recommendations(self, result: AIMSResult) -> list[str]:
        """Génère recommandations ISO 42001 Clause 10.2."""
        recommendations = []

        # Calibration recommendations
        if result.calibration:
            if result.calibration.improvement_pct < 0:
                recommendations.append(
                    "CALIBRATION: Dégradation détectée - considérer méthode Platt"
                )
            elif result.calibration.ece_after > 0.1:
                recommendations.append(
                    "CALIBRATION: ECE > 10% - recalibration recommandée"
                )

        # Uncertainty recommendations
        if result.uncertainty:
            if result.uncertainty.coverage < 0.85:
                recommendations.append(
                    f"UNCERTAINTY: Coverage {result.uncertainty.coverage:.1%} < 85% - "
                    "réduire alpha ou augmenter données calibration"
                )
            if result.uncertainty.empty_set_rate > 0.05:
                recommendations.append(
                    "UNCERTAINTY: >5% ensembles vides - vérifier qualité modèle"
                )

        # Alerting recommendations
        if result.alerting and not result.alerting.slack_configured:
            recommendations.append(
                "ALERTING: Configurer ALICE_SLACK_WEBHOOK pour monitoring production"
            )

        if not recommendations:
            recommendations.append("AIMMS: Tous les critères ISO 42001 satisfaits")

        return recommendations

    @property
    def calibrated_model(self) -> Any | None:
        """Retourne le modèle calibré (si calibration effectuée)."""
        if self._calibrator is None:
            return None
        # Note: Le calibrator stocke le résultat en interne
        return None  # TODO: Exposer le calibrated model

    @property
    def conformal_predictor(self) -> ConformalPredictor | None:
        """Retourne le prédicteur conformal (si uncertainty effectuée)."""
        return self._conformal


def run_postprocessing(
    model: Any,
    X_calib: np.ndarray | pd.DataFrame,
    y_calib: np.ndarray,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | None = None,
    model_version: str = "unknown",
    config: AIMSConfig | None = None,
) -> AIMSResult:
    """Fonction utilitaire pour postprocessing rapide.

    Args:
        model: Modèle entraîné
        X_calib: Features calibration
        y_calib: Labels calibration
        X_test: Features test
        y_test: Labels test (optionnel)
        model_version: Version modèle
        config: Configuration AIMMS

    Returns:
        AIMSResult
    """
    postprocessor = AIMSPostprocessor(config)
    return postprocessor.run(model, X_calib, y_calib, X_test, y_test, model_version)
