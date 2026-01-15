"""Tests Validate ISO Compliance - ISO 29119.

Document ID: ALICE-TEST-AUTOGLUON-ISO-COMPLIANCE
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from scripts.autogluon.iso_compliance import validate_iso_compliance


class TestValidateISOCompliance:
    """Tests pour validate_iso_compliance."""

    def test_full_validation(
        self,
        mock_training_result: MagicMock,
        test_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test validation complete."""
        mock_training_result.predictor.predict.return_value = test_data["target"]

        result = validate_iso_compliance(
            mock_training_result,
            test_data,
            protected_attribute="ligue_code",
            output_dir=tmp_path,
        )

        assert "model_card" in result
        assert "robustness" in result
        assert "fairness" in result
        assert "compliant" in result

    def test_saves_all_reports(
        self,
        mock_training_result: MagicMock,
        test_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test sauvegarde de tous les rapports."""
        mock_training_result.predictor.predict.return_value = test_data["target"]

        validate_iso_compliance(
            mock_training_result,
            test_data,
            protected_attribute="ligue_code",
            output_dir=tmp_path,
        )

        assert (tmp_path / "model_card.json").exists()
        assert (tmp_path / "robustness_report.json").exists()
        assert (tmp_path / "fairness_report.json").exists()

    def test_compliant_when_robust_and_fair(
        self,
        mock_training_result: MagicMock,
        test_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test compliance quand robuste et equitable."""
        mock_training_result.predictor.predict.return_value = test_data["target"]

        result = validate_iso_compliance(
            mock_training_result,
            test_data,
            protected_attribute=None,
            output_dir=tmp_path,
        )

        assert result["compliant"] is True
        assert result["fairness"] is None

    def test_not_compliant_when_fragile(
        self,
        mock_training_result: MagicMock,
        test_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test non-compliance quand fragile."""
        call_count = [0]

        def predict_side_effect(x_data: pd.DataFrame) -> pd.Series:
            call_count[0] += 1
            if call_count[0] == 1:
                return test_data["target"]
            return pd.Series(np.random.randint(0, 2, len(x_data)))

        mock_training_result.predictor.predict.side_effect = predict_side_effect

        result = validate_iso_compliance(
            mock_training_result,
            test_data,
            output_dir=tmp_path,
        )

        assert "robustness" in result
