"""Tests Cloud Promotion — ISO 29119/42001.

Document ID: ALICE-TEST-CLOUD-PROMOTE
Version: 1.0.0
Tests count: 6
Classes: TestHyperparamsSync, TestPromoteModel
"""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.kaggle_trainers import default_hyperparameters


class TestHyperparamsSync:
    """Tests config matches YAML -- 2 tests."""

    def test_kaggle_params_match_yaml(self) -> None:
        """Keys in default_hyperparameters() must match config/hyperparameters.yaml."""
        with Path("config/hyperparameters.yaml").open() as fh:
            yaml_cfg = yaml.safe_load(fh)
        kaggle_cfg = default_hyperparameters()
        skip_keys = {
            "thread_count",
            "n_jobs",
            "cat_features",
            "categorical_feature",
            "device",
            "task_type",
            "train_dir",
            "tree_method",
        }
        for section in ("catboost", "xgboost", "lightgbm"):
            yaml_keys = {k for k in yaml_cfg[section] if k not in skip_keys}
            kaggle_keys = {k for k in kaggle_cfg[section] if k not in skip_keys}
            assert yaml_keys == kaggle_keys, f"[{section}] mismatch"

    def test_multiclass_keys_present(self) -> None:
        """Multiclass-specific keys must exist in both YAML and code."""
        with Path("config/hyperparameters.yaml").open() as fh:
            yaml_cfg = yaml.safe_load(fh)
        kaggle_cfg = default_hyperparameters()
        # CatBoost: loss_function=MultiClass
        assert yaml_cfg["catboost"].get("loss_function") == "MultiClass"
        assert kaggle_cfg["catboost"].get("loss_function") == "MultiClass"
        # XGBoost: objective=multi:softprob, num_class=3
        assert yaml_cfg["xgboost"].get("objective") == "multi:softprob"
        assert kaggle_cfg["xgboost"].get("objective") == "multi:softprob"
        assert yaml_cfg["xgboost"].get("num_class") == 3
        assert kaggle_cfg["xgboost"].get("num_class") == 3
        # LightGBM: objective=multiclass, num_class=3
        assert yaml_cfg["lightgbm"].get("objective") == "multiclass"
        assert kaggle_cfg["lightgbm"].get("objective") == "multiclass"
        assert yaml_cfg["lightgbm"].get("num_class") == 3
        assert kaggle_cfg["lightgbm"].get("num_class") == 3


class TestPromoteModel:
    """Tests promotion logic -- 4 tests."""

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
