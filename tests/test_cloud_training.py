"""Tests Cloud Training — ISO 29119/42001.

Document ID: ALICE-TEST-CLOUD-TRAINING
Version: 3.0.0
Tests count: 18
Classes: TestComputeHash, TestBuildLineage, TestQualityGates, TestModelCard
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.kaggle_artifacts import (
    build_lineage,
    build_model_card,
    compute_dataframe_hash,
)
from scripts.kaggle_metrics import check_quality_gates
from scripts.kaggle_trainers import (
    LABEL_COLUMN,
    MODEL_EXTENSIONS,
    default_hyperparameters,
)


@pytest.fixture()
def small_df() -> pd.DataFrame:
    """Minimal DataFrame with LABEL_COLUMN for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "blanc_elo": np.random.randint(1200, 2200, 50),
            "noir_elo": np.random.randint(1200, 2200, 50),
            "diff_elo": np.random.randint(-400, 400, 50),
            LABEL_COLUMN: np.random.choice([0.0, 0.5, 1.0], 50),
        }
    )


@pytest.fixture()
def mock_results() -> dict:
    """Fake training results dict (multiclass 3-class)."""
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.15, 0.15, 0.70]] * 100)
    # fmt: off
    return {"CatBoost": {
        "model": m,
        "metrics": {"log_loss": 0.75, "accuracy_3class": 0.70, "test_log_loss": 0.74,
                     "test_accuracy": 0.70, "test_f1_macro": 0.68, "train_time_s": 1.0},
        "importance": {"blanc_elo": 0.5, "noir_elo": 0.3},
    }}
    # fmt: on


@pytest.fixture()
def mock_lineage() -> dict:
    """Fake lineage dict."""
    return {
        "train": {"path": "/data/train.parquet", "samples": 800, "hash": "abc123"},
        "valid": {"path": "/data/valid.parquet", "samples": 100, "hash": "def456"},
        "test": {"path": "/data/test.parquet", "samples": 100, "hash": "ghi789"},
        "feature_count": 10,
        "target_distribution": {
            "class_distribution": {"loss": 0.33, "draw": 0.33, "win": 0.34},
            "total_samples": 800,
        },
        "created_at": "2026-03-18T12:00:00+00:00",
    }


class TestComputeHash:
    """Tests dataframe hashing -- 2 tests."""

    def test_hash_deterministic(self, small_df: pd.DataFrame) -> None:
        """Same DataFrame must produce the same 16-char hex hash."""
        h1 = compute_dataframe_hash(small_df)
        h2 = compute_dataframe_hash(small_df.copy())
        assert h1 == h2
        assert len(h1) == 16
        assert all(c in "0123456789abcdef" for c in h1)

    def test_hash_differs_on_change(self, small_df: pd.DataFrame) -> None:
        """Modifying one value must produce a different hash."""
        h1 = compute_dataframe_hash(small_df)
        modified = small_df.copy()
        modified.iloc[0, 0] = modified.iloc[0, 0] + 9999
        assert h1 != compute_dataframe_hash(modified)


class TestBuildLineage:
    """Tests ISO 5259 lineage -- 2 tests."""

    def test_lineage_has_all_fields(self, small_df: pd.DataFrame, tmp_path: Path) -> None:
        """Lineage dict must match production nested format."""
        lineage = build_lineage(small_df, small_df, small_df, tmp_path)
        # Nested split entries (matching production metadata.json)
        for split in ("train", "valid", "test"):
            assert split in lineage, f"Missing split: {split}"
            assert "path" in lineage[split]
            assert "samples" in lineage[split]
            assert "hash" in lineage[split]
        assert lineage["train"]["samples"] == len(small_df)
        # Top-level fields
        assert "feature_count" in lineage
        assert "target_distribution" in lineage
        assert "created_at" in lineage
        assert "class_distribution" in lineage["target_distribution"]

    def test_lineage_feature_count_excludes_target(
        self, small_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """feature_count must equal len(columns) - 1 (target excluded)."""
        lineage = build_lineage(small_df, small_df, small_df, tmp_path)
        assert lineage["feature_count"] == len(small_df.columns) - 1


class TestQualityGates:
    """Tests 8-condition quality gates (multiclass) -- 6 tests."""

    def _make_results(self, ll: float, ece_draw: float = 0.01, bias: float = 0.005) -> dict:
        m = MagicMock()
        return {
            "CatBoost": {
                "model": m,
                "metrics": {
                    "test_log_loss": ll,
                    "test_accuracy": 0.7,
                    "test_f1_macro": 0.6,
                    "test_rps": 0.15,
                    "test_brier": 0.30,
                    "test_es_mae": 0.25,
                    "ece_class_loss": 0.01,
                    "ece_class_draw": ece_draw,
                    "ece_class_win": 0.01,
                    "draw_calibration_bias": bias,
                    "recall_draw": 0.05,
                    "mean_p_draw": 0.14,
                },
                "importance": {},
            }
        }

    def _make_baselines(self, naive_ll: float = 1.10, elo_ll: float = 1.05) -> dict:
        return {
            "naive": {"log_loss": naive_ll, "rps": 0.22, "brier": 0.40},
            "elo": {"log_loss": elo_ll, "rps": 0.20, "es_mae": 0.35},
        }

    def test_logloss_beats_naive_fails(self) -> None:
        """LogLoss 1.20 >= naive 1.10 must fail quality gate."""
        gate = check_quality_gates(
            self._make_results(1.20), baseline_metrics=self._make_baselines()
        )
        assert gate["passed"] is False
        assert "naive" in gate["reason"]

    def test_logloss_beats_baselines_passes(self) -> None:
        """LogLoss 0.85 < naive and elo must pass baseline checks."""
        gate = check_quality_gates(
            self._make_results(0.85), baseline_metrics=self._make_baselines()
        )
        assert gate["passed"] is True
        assert gate["best_log_loss"] == pytest.approx(0.85)

    def test_ece_draw_above_threshold_fails(self) -> None:
        """ECE draw 0.08 >= 0.05 must fail quality gate."""
        gate = check_quality_gates(
            self._make_results(0.85, ece_draw=0.08), baseline_metrics=self._make_baselines()
        )
        assert gate["passed"] is False
        assert "ece_draw" in gate["reason"]

    def test_draw_calibration_bias_fails(self) -> None:
        """draw_calibration_bias 0.03 >= 0.02 must fail quality gate."""
        gate = check_quality_gates(
            self._make_results(0.85, bias=0.03), baseline_metrics=self._make_baselines()
        )
        assert gate["passed"] is False
        assert "draw_calibration_bias" in gate["reason"]

    def test_degradation_relative_fails(self) -> None:
        """Champion 0.80, new 0.85 -> 6.25% rise > 5% threshold -> fail."""
        gate = check_quality_gates(
            self._make_results(0.85), baseline_metrics=self._make_baselines(), champion_ll=0.80
        )
        assert gate["passed"] is False
        assert "Degradation" in gate["reason"]

    def test_first_run_no_champion_passes(self) -> None:
        """champion_ll=None must skip degradation check."""
        gate = check_quality_gates(
            self._make_results(0.85), baseline_metrics=self._make_baselines(), champion_ll=None
        )
        assert gate["passed"] is True

    def test_t9_few_features_fails(self) -> None:
        """T9: model with <=5 features importance>0 must fail."""
        results = self._make_results(0.85)
        results["CatBoost"]["model"].feature_importances_ = np.array([0.5, 0.3, 0.1, 0.0, 0.0])
        gate = check_quality_gates(
            results,
            baseline_metrics=self._make_baselines(),
            verbose=False,
        )
        assert gate["passed"] is False
        assert "T9" in gate["reason"]

    def test_t9_many_features_passes(self) -> None:
        """T9: model with >5 features importance>0 must pass."""
        results = self._make_results(0.85)
        results["CatBoost"]["model"].feature_importances_ = np.array(
            [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.0, 0.0, 0.0]
        )
        gate = check_quality_gates(
            results,
            baseline_metrics=self._make_baselines(),
            verbose=False,
        )
        assert gate["passed"] is True

    def test_t10_small_gap_passes(self) -> None:
        """T10: train-test gap < 0.05 must pass (no warning)."""
        gate = check_quality_gates(
            self._make_results(0.85),
            baseline_metrics=self._make_baselines(),
            train_log_loss=0.82,
            verbose=False,
        )
        assert gate["passed"] is True

    def test_t10_large_gap_warns_but_passes(self) -> None:
        """T10: train-test gap > 0.05 logs warning but still passes (temporal splits)."""
        gate = check_quality_gates(
            self._make_results(0.85),
            baseline_metrics=self._make_baselines(),
            train_log_loss=0.75,
            verbose=False,
        )
        # T10 is WARNING, not FAIL — expected for temporal splits
        assert gate["passed"] is True


class TestModelCard:
    """Tests model card structure -- 4 tests."""

    @pytest.fixture()
    def gate(self) -> dict:
        return {"passed": True, "best_model": "CatBoost", "best_log_loss": 0.85}

    def test_card_has_all_required_fields(
        self, mock_results: dict, mock_lineage: dict, gate: dict
    ) -> None:
        """Model card dict must contain all fields including quality_gate_result."""
        card = build_model_card(
            mock_results, mock_lineage, gate, default_hyperparameters(), MODEL_EXTENSIONS
        )
        required_fields = [
            "version",
            "created_at",
            "status",
            "environment",
            "data_lineage",
            "artifacts",
            "metrics",
            "feature_importance",
            "hyperparameters",
            "best_model",
            "quality_gate_result",
            "limitations",
            "use_cases",
            "conformance",
        ]
        for f in required_fields:
            assert f in card, f"Missing field: {f}"

    def test_card_artifacts_have_checksums(
        self, mock_results: dict, mock_lineage: dict, gate: dict
    ) -> None:
        """Each artifact entry must have sha256 and size_bytes fields."""
        card = build_model_card(
            mock_results, mock_lineage, gate, default_hyperparameters(), MODEL_EXTENSIONS
        )
        for artifact in card["artifacts"]:
            assert "sha256" in artifact, f"Missing sha256 in {artifact}"
            assert "size_bytes" in artifact, f"Missing size_bytes in {artifact}"

    def test_card_environment_has_versions(
        self, mock_results: dict, mock_lineage: dict, gate: dict
    ) -> None:
        """Environment must include python_version, catboost, xgboost, lightgbm."""
        card = build_model_card(
            mock_results, mock_lineage, gate, default_hyperparameters(), MODEL_EXTENSIONS
        )
        env = card["environment"]
        assert "python_version" in env
        for pkg in ("catboost", "xgboost", "lightgbm"):
            assert pkg in env.get("packages", {}), f"Missing package: {pkg}"

    def test_card_has_quality_gate_result(
        self, mock_results: dict, mock_lineage: dict, gate: dict
    ) -> None:
        """Model card must contain quality_gate_result matching the gate dict."""
        card = build_model_card(
            mock_results, mock_lineage, gate, default_hyperparameters(), MODEL_EXTENSIONS
        )
        assert card["quality_gate_result"] == gate
