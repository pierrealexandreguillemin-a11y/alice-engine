"""Tests Ensemble Model Factory - ISO 29119/42001.

Document ID: ALICE-TEST-FACTORY
Version: 1.0.0
Tests: 7

Classes:
- TestCreateCatboostModel: Tests création CatBoost (2 tests)
- TestCreateXgboostModel: Tests création XGBoost (2 tests)
- TestCreateLightgbmModel: Tests création LightGBM (2 tests)
- TestCreateModelByName: Tests factory par nom (1 test)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management
- ISO/IEC 5055:2021 - Code Quality (<80 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import pytest

from scripts.ensemble.model_factory import (
    create_catboost_model,
    create_lightgbm_model,
    create_model_by_name,
    create_xgboost_model,
)


class TestCreateCatboostModel:
    """Tests pour create_catboost_model."""

    def test_creates_classifier(self):
        """Crée un CatBoostClassifier."""
        model = create_catboost_model({"iterations": 10}, [0])

        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_applies_params(self):
        """Applique les paramètres fournis."""
        model = create_catboost_model(
            {"iterations": 50, "depth": 4, "learning_rate": 0.1},
            [],
        )

        assert model.get_params()["iterations"] == 50
        assert model.get_params()["depth"] == 4


class TestCreateXgboostModel:
    """Tests pour create_xgboost_model."""

    def test_creates_classifier(self):
        """Crée un XGBClassifier."""
        model = create_xgboost_model({"n_estimators": 10})

        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_applies_params(self):
        """Applique les paramètres."""
        model = create_xgboost_model({"n_estimators": 50, "max_depth": 4})

        assert model.get_params()["n_estimators"] == 50
        assert model.get_params()["max_depth"] == 4


class TestCreateLightgbmModel:
    """Tests pour create_lightgbm_model."""

    def test_creates_classifier(self):
        """Crée un LGBMClassifier."""
        model = create_lightgbm_model({"n_estimators": 10}, [])

        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_applies_params(self):
        """Applique les paramètres."""
        model = create_lightgbm_model({"n_estimators": 50, "num_leaves": 31}, [])

        assert model.get_params()["n_estimators"] == 50
        assert model.get_params()["num_leaves"] == 31


class TestCreateModelByName:
    """Tests pour create_model_by_name."""

    def test_unknown_model_raises(self):
        """Lève erreur pour modèle inconnu."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_model_by_name("UnknownModel", {}, [])
