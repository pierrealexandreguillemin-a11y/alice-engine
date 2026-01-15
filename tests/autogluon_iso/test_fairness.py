"""Tests Fairness ISO 24027 - ISO 29119.

Document ID: ALICE-TEST-AUTOGLUON-ISO-FAIR
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC TR 24027:2021 - Bias Detection
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from scripts.autogluon.iso_compliance import (
    ISO24027BiasReport,
    validate_fairness,
)


class TestISO24027BiasReport:
    """Tests pour ISO24027BiasReport."""

    def test_dataclass_creation(self) -> None:
        """Test creation du dataclass."""
        report = ISO24027BiasReport(
            model_id="test",
            demographic_parity=0.85,
            equalized_odds=0.9,
            calibration_by_group={"A": 0.8, "B": 0.82},
            status="FAIR",
        )

        assert report.status == "FAIR"
        assert report.demographic_parity == 0.85

    def test_default_values(self) -> None:
        """Test valeurs par defaut."""
        report = ISO24027BiasReport(
            model_id="test",
            demographic_parity=0.0,
            equalized_odds=0.0,
        )

        assert report.calibration_by_group == {}
        assert report.status == "NOT_EVALUATED"


class TestValidateFairness:
    """Tests pour validate_fairness."""

    def test_fair_model(
        self,
        mock_predictor: MagicMock,
        test_data: pd.DataFrame,
    ) -> None:
        """Test modele equitable."""
        mock_predictor.predict.return_value = pd.Series(
            np.ones(len(test_data), dtype=int),
        )

        report = validate_fairness(
            mock_predictor,
            test_data,
            protected_attribute="ligue_code",
        )

        assert isinstance(report, ISO24027BiasReport)
        assert report.demographic_parity == 1.0
        assert report.status == "FAIR"

    def test_biased_model(
        self,
        mock_predictor: MagicMock,
        test_data: pd.DataFrame,
    ) -> None:
        """Test modele biaise."""
        predictions = np.where(test_data["ligue_code"] == "IDF", 1, 0)
        mock_predictor.predict.return_value = pd.Series(predictions)

        report = validate_fairness(
            mock_predictor,
            test_data,
            protected_attribute="ligue_code",
        )

        assert report.demographic_parity < 0.8
        assert report.status in ["CAUTION", "CRITICAL"]

    def test_calibration_by_group(
        self,
        mock_predictor: MagicMock,
        test_data: pd.DataFrame,
    ) -> None:
        """Test calibration par groupe."""
        predictions = np.where(test_data["ligue_code"] == "IDF", 1, 0)
        mock_predictor.predict.return_value = pd.Series(predictions)

        report = validate_fairness(
            mock_predictor,
            test_data,
            protected_attribute="ligue_code",
        )

        assert "IDF" in report.calibration_by_group
        assert report.calibration_by_group["IDF"] > 0
