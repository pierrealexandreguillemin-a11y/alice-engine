"""Tests Cloud Training - ISO 29119.

Document ID: ALICE-TEST-CLOUD-TRAINING
Version: 2.0.0
Tests: 16

Classes:
- TestComputeHash: Tests dataframe hashing (2 tests)
- TestBuildLineage: Tests ISO 5259 lineage (2 tests)
- TestQualityGates: Tests AUC gates (4 tests)
- TestModelCard: Tests model card structure (4 tests)
- TestHyperparamsSync: Tests config matches YAML (1 test)
- TestPromoteModel: Tests promotion logic (3 tests)

ISO: 29119, 42001. Author: ALICE Engine Team. Updated: 2026-03-18
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.cloud.train_kaggle import (
    LABEL_COLUMN,
    build_lineage,
    build_model_card,
    check_quality_gates,
    compute_dataframe_hash,
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
    """Fake training results dict."""
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.3, 0.7]] * 100)
    # fmt: off
    return {"CatBoost": {
        "model": m,
        "metrics": {"auc_roc": 0.75, "accuracy": 0.70, "f1_score": 0.68,
                     "test_auc": 0.74, "test_accuracy": 0.70, "test_f1": 0.68, "train_time_s": 1.0},
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
        "target_distribution": {"positive_ratio": 0.5, "total_samples": 800},
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
        assert "positive_ratio" in lineage["target_distribution"]

    def test_lineage_feature_count_excludes_target(
        self, small_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """feature_count must equal len(columns) - 1 (target excluded)."""
        lineage = build_lineage(small_df, small_df, small_df, tmp_path)
        assert lineage["feature_count"] == len(small_df.columns) - 1


class TestQualityGates:
    """Tests AUC quality gates -- 4 tests."""

    def _make_results(self, auc: float) -> dict:
        m = MagicMock()
        return {
            "CatBoost": {
                "model": m,
                "metrics": {"test_auc": auc, "test_accuracy": 0.7, "test_f1": 0.6},
                "importance": {},
            }
        }

    def test_auc_below_floor_fails(self) -> None:
        """AUC 0.65 < AUC_FLOOR must fail quality gate."""
        gate = check_quality_gates(self._make_results(0.65))
        assert gate["passed"] is False
        assert "AUC" in gate["reason"]

    def test_auc_above_floor_passes(self) -> None:
        """AUC 0.75 > AUC_FLOOR must pass quality gate."""
        gate = check_quality_gates(self._make_results(0.75))
        assert gate["passed"] is True
        assert gate["best_auc"] == pytest.approx(0.75)

    def test_degradation_relative_fails(self) -> None:
        """Champion 0.75, new 0.72 -> 4% drop > 2% threshold -> fail."""
        gate = check_quality_gates(self._make_results(0.72), champion_auc=0.75)
        assert gate["passed"] is False
        assert "Degradation" in gate["reason"]

    def test_first_run_no_champion_passes(self) -> None:
        """champion_auc=None must skip degradation check."""
        gate = check_quality_gates(self._make_results(0.75), champion_auc=None)
        assert gate["passed"] is True


class TestModelCard:
    """Tests model card structure -- 4 tests."""

    @pytest.fixture()
    def gate(self) -> dict:
        return {"passed": True, "best_model": "CatBoost", "best_auc": 0.74}

    def test_card_has_all_required_fields(
        self, mock_results: dict, mock_lineage: dict, gate: dict
    ) -> None:
        """Model card dict must contain all fields including quality_gate_result."""
        card = build_model_card(mock_results, mock_lineage, gate, default_hyperparameters())
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
        card = build_model_card(mock_results, mock_lineage, gate, default_hyperparameters())
        for artifact in card["artifacts"]:
            assert "sha256" in artifact, f"Missing sha256 in {artifact}"
            assert "size_bytes" in artifact, f"Missing size_bytes in {artifact}"

    def test_card_environment_has_versions(
        self, mock_results: dict, mock_lineage: dict, gate: dict
    ) -> None:
        """Environment must include python_version, catboost, xgboost, lightgbm."""
        card = build_model_card(mock_results, mock_lineage, gate, default_hyperparameters())
        env = card["environment"]
        assert "python_version" in env
        for pkg in ("catboost", "xgboost", "lightgbm"):
            assert pkg in env.get("packages", {}), f"Missing package: {pkg}"

    def test_card_has_quality_gate_result(
        self, mock_results: dict, mock_lineage: dict, gate: dict
    ) -> None:
        """Model card must contain quality_gate_result matching the gate dict."""
        card = build_model_card(mock_results, mock_lineage, gate, default_hyperparameters())
        assert card["quality_gate_result"] == gate


class TestHyperparamsSync:
    """Tests config matches YAML -- 1 test."""

    def test_kaggle_params_match_yaml(self) -> None:
        """Keys in default_hyperparameters() must match config/hyperparameters.yaml."""
        with Path("config/hyperparameters.yaml").open() as fh:
            yaml_cfg = yaml.safe_load(fh)
        kaggle_cfg = default_hyperparameters()
        skip_keys = {"thread_count", "n_jobs", "cat_features", "categorical_feature"}
        for section in ("catboost", "xgboost", "lightgbm"):
            yaml_keys = {k for k in yaml_cfg[section] if k not in skip_keys}
            kaggle_keys = {k for k in kaggle_cfg[section] if k not in skip_keys}
            assert yaml_keys == kaggle_keys, f"[{section}] mismatch"


class TestPromoteModel:
    """Tests promotion logic -- 3 tests."""

    def _fake_robustness(self, compliant: bool) -> dict:
        return {
            "base_auc": 0.74,
            "noisy_auc": 0.73 if compliant else 0.50,
            "noise_tolerance": 0.99 if compliant else 0.67,
            "compliant": compliant,
        }

    def _fake_fairness(self, status: str) -> dict:
        return {"status": status, "demographic_parity": 0.9, "group_rates": {}}

    def _mcnemar(self) -> dict:
        return {"p_value": 0.5, "significant": False, "new_auc": 0.74}

    def test_robustness_fail_rejects(self) -> None:
        """Non-compliant robustness must result in REJECTED status."""
        from scripts.cloud.promote_model import decide_promotion

        result = decide_promotion(
            self._fake_robustness(False), self._fake_fairness("FAIR"), self._mcnemar()
        )
        assert result["decision"] == "REJECTED"
        assert "robustness" in result["reason"].lower()

    def test_fairness_critical_rejects(self) -> None:
        """CRITICAL fairness must result in REJECTED status."""
        from scripts.cloud.promote_model import decide_promotion

        result = decide_promotion(
            self._fake_robustness(True), self._fake_fairness("CRITICAL"), self._mcnemar()
        )
        assert result["decision"] == "REJECTED"
        assert "fairness" in result["reason"].lower()

    def test_mcnemar_significantly_worse_rejects(self) -> None:
        """Significantly worse on McNemar must result in REJECTED."""
        from scripts.cloud.promote_model import decide_promotion

        mcnemar_worse = {
            "p_value": 0.001,
            "significant": True,
            "new_auc": 0.70,
            "champion_auc": 0.75,
        }
        result = decide_promotion(
            self._fake_robustness(True),
            self._fake_fairness("FAIR"),
            mcnemar_worse,
        )
        assert result["decision"] == "REJECTED"
        assert "mcnemar" in result["reason"].lower()

    def test_all_pass_promotes(self) -> None:
        """All checks passing must result in PRODUCTION status."""
        from scripts.cloud.promote_model import decide_promotion

        result = decide_promotion(
            self._fake_robustness(True), self._fake_fairness("FAIR"), self._mcnemar()
        )
        assert result["decision"] == "PRODUCTION"
