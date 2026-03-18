"""Tests Cloud Features & Metrics - ISO 29119.

Document ID: ALICE-TEST-CLOUD-FEATURES
Version: 1.0.0
Tests: 7

Classes:
- TestPrepareFeatures: Tests feature preparation (3 tests)
- TestComputeValidationMetrics: Tests metrics computation (2 tests)
- TestFetchChampionAuc: Tests champion AUC fetching (2 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 5055:2021 - Code Quality

Author: ALICE Engine Team
Last Updated: 2026-03-18
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.cloud.train_kaggle import (
    CATBOOST_CAT_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL_COLUMN,
    compute_validation_metrics,
    prepare_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def realistic_df() -> pd.DataFrame:
    """DataFrame with string columns mimicking real data for prepare_features."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "blanc_elo": rng.integers(1200, 2200, n),
            "noir_elo": rng.integers(1200, 2200, n),
            "diff_elo": rng.integers(-400, 400, n),
            "echiquier": rng.integers(1, 8, n),
            "niveau": rng.integers(1, 5, n),
            "ronde": rng.integers(1, 11, n),
            "type_competition": rng.choice(["interclub", "autre"], n),
            "division": rng.choice(["N1", "N2", "N3"], n),
            "ligue_code": rng.choice(["IDF", "ARA", "OCC"], n),
            "jour_semaine": rng.choice(["Samedi", "Dimanche"], n),
            "blanc_titre": rng.choice(["GM", "IM", ""], n),
            "noir_titre": rng.choice(["GM", "FM", ""], n),
            "zone_enjeu_dom": rng.choice(["danger", "confort"], n),
            "blanc_nom": [f"Player_{i}" for i in range(n)],
            "noir_nom": [f"Opponent_{i}" for i in range(n)],
            "equipe_dom": [f"Team_{i % 10}" for i in range(n)],
            "resultat_text": rng.choice(["1-0", "0-1", "0.5-0.5"], n),
            LABEL_COLUMN: rng.choice([0.0, 0.5, 1.0], n),
            "resultat_noir": rng.choice([0.0, 0.5, 1.0], n),
        }
    )


# ---------------------------------------------------------------------------
# TestPrepareFeatures
# ---------------------------------------------------------------------------


class TestPrepareFeatures:
    """Tests feature preparation — 3 tests."""

    def test_no_string_columns_in_x(self, realistic_df: pd.DataFrame) -> None:
        """After prepare_features, X must contain zero object/category columns."""
        X_train, _, X_valid, _, X_test, _, _ = prepare_features(
            realistic_df.copy(), realistic_df.copy(), realistic_df.copy()
        )
        for label, X in [("train", X_train), ("valid", X_valid), ("test", X_test)]:
            obj_cols = X.select_dtypes(include=["object", "category"]).columns
            assert len(obj_cols) == 0, f"{label} has string cols: {list(obj_cols)}"

    def test_all_categoricals_encoded(self, realistic_df: pd.DataFrame) -> None:
        """All CATEGORICAL + CATBOOST_CAT features must be label-encoded (numeric)."""
        X_train, _, _, _, _, _, encoders = prepare_features(
            realistic_df.copy(), realistic_df.copy(), realistic_df.copy()
        )
        all_cats = sorted(set(CATEGORICAL_FEATURES) | set(CATBOOST_CAT_FEATURES))
        for col in all_cats:
            if col in X_train.columns:
                assert X_train[col].dtype in (
                    np.int32,
                    np.int64,
                    np.float64,
                ), f"{col} dtype={X_train[col].dtype} is not numeric"
            assert col in encoders, f"Missing encoder for {col}"

    def test_target_columns_dropped(self, realistic_df: pd.DataFrame) -> None:
        """Target columns must not appear in X."""
        X_train, _, _, _, _, _, _ = prepare_features(
            realistic_df.copy(), realistic_df.copy(), realistic_df.copy()
        )
        forbidden = [LABEL_COLUMN, "resultat_noir", "resultat_text"]
        for col in forbidden:
            assert col not in X_train.columns, f"{col} should be dropped"


# ---------------------------------------------------------------------------
# TestComputeValidationMetrics
# ---------------------------------------------------------------------------


class TestComputeValidationMetrics:
    """Tests metrics computation — 2 tests."""

    @pytest.fixture()
    def binary_predictions(self) -> tuple:
        """Generate deterministic binary predictions for metrics testing."""
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1], 200)
        y_proba = np.where(y_true == 1, rng.uniform(0.5, 0.9, 200), rng.uniform(0.1, 0.5, 200))
        y_pred = (y_proba >= 0.5).astype(int)
        return y_true, y_pred, y_proba

    def test_metrics_has_10_fields(self, binary_predictions: tuple) -> None:
        """compute_validation_metrics must return exactly 10 fields."""
        y_true, y_pred, y_proba = binary_predictions
        metrics = compute_validation_metrics(y_true, y_pred, y_proba)
        expected_keys = {
            "auc_roc",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "log_loss",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
        }
        assert set(metrics.keys()) == expected_keys

    def test_metrics_values_in_range(self, binary_predictions: tuple) -> None:
        """All probability metrics must be in [0, 1], confusion counts >= 0."""
        y_true, y_pred, y_proba = binary_predictions
        metrics = compute_validation_metrics(y_true, y_pred, y_proba)
        for key in ("auc_roc", "accuracy", "precision", "recall", "f1_score"):
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of range"
        assert metrics["log_loss"] >= 0.0
        for key in ("true_negatives", "false_positives", "false_negatives", "true_positives"):
            assert metrics[key] >= 0, f"{key}={metrics[key]} negative"


# ---------------------------------------------------------------------------
# TestFetchChampionAuc
# ---------------------------------------------------------------------------


class TestFetchChampionAuc:
    """Tests champion AUC fetching — 2 tests."""

    def test_reads_best_model_auc(self, tmp_path: object) -> None:
        """fetch_champion_auc must read best_model.auc from metadata.json."""
        metadata = {"best_model": {"name": "CatBoost", "auc": 0.7533}}
        fake_path = str(tmp_path) + "/metadata.json"  # type: ignore[operator]
        with open(fake_path, "w") as f:
            json.dump(metadata, f)

        mock_download = MagicMock(return_value=fake_path)
        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(hf_hub_download=mock_download)},
        ):
            # Re-import to pick up mocked module
            import importlib

            import scripts.cloud.train_kaggle as mod

            importlib.reload(mod)
            result = mod.fetch_champion_auc()
            # Restore module
            importlib.reload(mod)
        assert result == pytest.approx(0.7533)

    def test_returns_none_on_failure(self) -> None:
        """fetch_champion_auc returns None when HF download fails."""
        mock_download = MagicMock(side_effect=Exception("Network error"))
        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(hf_hub_download=mock_download)},
        ):
            import importlib

            import scripts.cloud.train_kaggle as mod

            importlib.reload(mod)
            result = mod.fetch_champion_auc()
            importlib.reload(mod)
        assert result is None
