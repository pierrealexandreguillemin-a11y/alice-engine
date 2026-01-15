"""Tests Robustness ISO 24029 - ISO 29119.

Document ID: ALICE-TEST-AUTOGLUON-ISO-ROBUST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from scripts.autogluon.iso_compliance import (
    ISO24029RobustnessReport,
    validate_robustness,
)


class TestISO24029RobustnessReport:
    """Tests pour ISO24029RobustnessReport."""

    def test_dataclass_creation(self) -> None:
        """Test creation du dataclass."""
        report = ISO24029RobustnessReport(
            model_id="test",
            noise_tolerance=0.95,
            adversarial_robustness=0.9,
            distribution_shift_score=0.85,
            confidence_calibration=0.92,
            status="ROBUST",
        )

        assert report.status == "ROBUST"
        assert report.noise_tolerance == 0.95

    def test_default_status(self) -> None:
        """Test statut par defaut."""
        report = ISO24029RobustnessReport(
            model_id="test",
            noise_tolerance=0.0,
            adversarial_robustness=0.0,
            distribution_shift_score=0.0,
            confidence_calibration=0.0,
        )

        assert report.status == "NOT_EVALUATED"


class TestValidateRobustness:
    """Tests pour validate_robustness."""

    def test_robust_model(
        self,
        mock_predictor: MagicMock,
        test_data: pd.DataFrame,
    ) -> None:
        """Test modele robuste."""
        mock_predictor.predict.return_value = test_data["target"]

        report = validate_robustness(mock_predictor, test_data)

        assert isinstance(report, ISO24029RobustnessReport)
        assert report.noise_tolerance >= 0.95
        assert report.status == "ROBUST"

    def test_sensitive_model(
        self,
        mock_predictor: MagicMock,
        test_data: pd.DataFrame,
    ) -> None:
        """Test modele sensible au bruit."""
        call_count = [0]

        def predict_side_effect(x_data: pd.DataFrame) -> pd.Series:
            call_count[0] += 1
            if call_count[0] == 1:
                return test_data["target"]
            return pd.Series(np.random.randint(0, 2, len(x_data)))

        mock_predictor.predict.side_effect = predict_side_effect

        report = validate_robustness(mock_predictor, test_data, noise_level=0.1)

        assert report.noise_tolerance < 0.95
        assert report.status in ["SENSITIVE", "FRAGILE"]

    def test_returns_model_id(
        self,
        mock_predictor: MagicMock,
        test_data: pd.DataFrame,
    ) -> None:
        """Test que model_id est retourne."""
        mock_predictor.predict.return_value = test_data["target"]

        report = validate_robustness(mock_predictor, test_data)

        assert report.model_id == "models/test"
