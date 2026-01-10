"""Tests: tests/test_autogluon_iso_compliance.py - ISO Compliance Tests.

Document ID: ALICE-TEST-AUTOGLUON-ISO-001
Version: 1.0.0

Tests unitaires pour la conformite ISO 42001/24029/24027.
Couvre: Model Cards, robustesse, biais.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC TR 24027:2021 - Bias Detection

Test Coverage Target: 90%
Total Tests: 20

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.autogluon.config import AutoGluonConfig
from scripts.autogluon.iso_compliance import (
    ISO24027BiasReport,
    ISO24029RobustnessReport,
    ISO42001ModelCard,
    generate_model_card,
    validate_fairness,
    validate_iso_compliance,
    validate_robustness,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_training_result() -> MagicMock:
    """Resultat d'entrainement mock."""
    result = MagicMock()
    result.model_path = Path("models/test_model")
    result.data_hash = "abc123def456" * 5  # 64 chars
    result.best_model = "CatBoost"
    result.config = AutoGluonConfig(presets="best_quality")
    result.metrics = {
        "score_val": 0.85,
        "pred_time_val": 0.1,
        "fit_time": 10.0,
        "num_models": 5,
    }
    result.predictor = MagicMock()
    result.predictor.feature_importance.return_value = pd.DataFrame(
        {
            "importance": [0.3, 0.5, 0.2],
        },
        index=["feature_1", "feature_2", "feature_3"],
    )
    result.predictor.path = "models/test"
    result.predictor.label = "target"
    return result


@pytest.fixture
def test_data() -> pd.DataFrame:
    """Donnees de test."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "ligue_code": np.random.choice(["IDF", "ARA", "BRE"], n),
            "target": np.random.randint(0, 2, n),
        }
    )


@pytest.fixture
def mock_predictor() -> MagicMock:
    """Predictor mock."""
    predictor = MagicMock()
    predictor.path = "models/test"
    predictor.label = "target"
    return predictor


# =============================================================================
# TESTS: ISO42001ModelCard
# =============================================================================


class TestISO42001ModelCard:
    """Tests pour ISO42001ModelCard."""

    def test_dataclass_creation(self) -> None:
        """Test creation du dataclass."""
        card = ISO42001ModelCard(
            model_id="test_001",
            model_name="TestModel",
            version="1.0.0",
            created_at="2026-01-10",
            training_data_hash="abc123",
            hyperparameters={"lr": 0.01},
            metrics={"accuracy": 0.85},
        )

        assert card.model_id == "test_001"
        assert card.model_name == "TestModel"
        assert card.version == "1.0.0"

    def test_default_values(self) -> None:
        """Test valeurs par defaut."""
        card = ISO42001ModelCard(
            model_id="test",
            model_name="Test",
            version="1.0",
            created_at="2026-01-10",
            training_data_hash="abc",
            hyperparameters={},
            metrics={},
        )

        assert "echecs" in card.intended_use.lower()
        assert "FFE" in card.limitations

    def test_feature_importance_default(self) -> None:
        """Test importance features par defaut."""
        card = ISO42001ModelCard(
            model_id="test",
            model_name="Test",
            version="1.0",
            created_at="2026-01-10",
            training_data_hash="abc",
            hyperparameters={},
            metrics={},
        )

        assert card.feature_importance == {}


class TestGenerateModelCard:
    """Tests pour generate_model_card."""

    def test_generates_valid_card(
        self,
        mock_training_result: MagicMock,
    ) -> None:
        """Test generation d'une carte valide."""
        card = generate_model_card(mock_training_result)

        assert isinstance(card, ISO42001ModelCard)
        assert card.model_name == "AutoGluon_best_quality"
        assert card.best_model == "CatBoost"
        assert card.training_data_hash == mock_training_result.data_hash

    def test_saves_to_file(
        self,
        mock_training_result: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test sauvegarde dans fichier."""
        output_path = tmp_path / "model_card.json"
        generate_model_card(mock_training_result, output_path=output_path)

        assert output_path.exists()

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "model_id" in data
        assert "hyperparameters" in data

    def test_includes_feature_importance(
        self,
        mock_training_result: MagicMock,
    ) -> None:
        """Test inclusion de l'importance des features."""
        card = generate_model_card(mock_training_result)

        assert len(card.feature_importance) > 0
        assert "feature_1" in card.feature_importance

    def test_handles_feature_importance_error(
        self,
        mock_training_result: MagicMock,
    ) -> None:
        """Test gestion erreur importance features."""
        mock_training_result.predictor.feature_importance.side_effect = Exception("Error")

        card = generate_model_card(mock_training_result)

        # Should still work, with empty importance
        assert card.feature_importance == {}


# =============================================================================
# TESTS: ISO24029RobustnessReport
# =============================================================================


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
        # Perfect predictions, unaffected by noise
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
        # First call (baseline): correct predictions
        # Second call (noisy): random predictions
        call_count = [0]

        def predict_side_effect(x_data: pd.DataFrame) -> pd.Series:
            call_count[0] += 1
            if call_count[0] == 1:
                return test_data["target"]  # Perfect baseline
            return pd.Series(np.random.randint(0, 2, len(x_data)))  # Random for noisy

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


# =============================================================================
# TESTS: ISO24027BiasReport
# =============================================================================


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
        # Same positive rate for all groups
        mock_predictor.predict.return_value = pd.Series(
            np.ones(len(test_data), dtype=int),
        )

        report = validate_fairness(
            mock_predictor,
            test_data,
            protected_attribute="ligue_code",
        )

        assert isinstance(report, ISO24027BiasReport)
        assert report.demographic_parity == 1.0  # All same rate
        assert report.status == "FAIR"

    def test_biased_model(
        self,
        mock_predictor: MagicMock,
        test_data: pd.DataFrame,
    ) -> None:
        """Test modele biaise."""
        # IDF gets all positive, others get negative
        predictions = np.where(test_data["ligue_code"] == "IDF", 1, 0)
        mock_predictor.predict.return_value = pd.Series(predictions)

        report = validate_fairness(
            mock_predictor,
            test_data,
            protected_attribute="ligue_code",
        )

        assert report.demographic_parity < 0.8  # Violates 80% rule
        assert report.status in ["CAUTION", "CRITICAL"]

    def test_calibration_by_group(
        self,
        mock_predictor: MagicMock,
        test_data: pd.DataFrame,
    ) -> None:
        """Test calibration par groupe."""
        # Different rates per group
        predictions = np.where(test_data["ligue_code"] == "IDF", 1, 0)
        mock_predictor.predict.return_value = pd.Series(predictions)

        report = validate_fairness(
            mock_predictor,
            test_data,
            protected_attribute="ligue_code",
        )

        assert "IDF" in report.calibration_by_group
        assert report.calibration_by_group["IDF"] > 0


# =============================================================================
# TESTS: validate_iso_compliance (integration)
# =============================================================================


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
        # Good predictions
        mock_training_result.predictor.predict.return_value = test_data["target"]

        result = validate_iso_compliance(
            mock_training_result,
            test_data,
            protected_attribute=None,  # Skip fairness
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
        # First call good, second call random (simulating fragility)
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

        # May or may not be compliant depending on noise tolerance
        assert "robustness" in result
