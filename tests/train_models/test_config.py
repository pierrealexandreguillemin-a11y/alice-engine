"""Tests Config - ISO 29119.

Document ID: ALICE-TEST-TRAIN-CONFIG
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

import yaml

from scripts.training import get_default_hyperparameters, load_hyperparameters


class TestLoadHyperparameters:
    """Tests pour load_hyperparameters."""

    def test_load_existing_config(self, tmp_path: Path) -> None:
        """Test chargement config existante."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "global": {"random_seed": 42},
            "catboost": {"iterations": 100},
        }
        with config_path.open("w") as f:
            yaml.dump(config_data, f)

        result = load_hyperparameters(config_path)

        assert result["global"] == {"random_seed": 42}
        assert result["catboost"] == {"iterations": 100}

    def test_load_nonexistent_config_returns_defaults(self, tmp_path: Path) -> None:
        """Test config inexistante retourne defaults."""
        config_path = tmp_path / "nonexistent.yaml"

        result = load_hyperparameters(config_path)

        assert "global" in result
        assert "catboost" in result
        assert "xgboost" in result
        assert "lightgbm" in result


class TestGetDefaultHyperparameters:
    """Tests pour get_default_hyperparameters."""

    def test_returns_all_model_configs(self) -> None:
        """Test retourne config pour tous les modeles."""
        defaults = get_default_hyperparameters()

        assert "global" in defaults
        assert "catboost" in defaults
        assert "xgboost" in defaults
        assert "lightgbm" in defaults

    def test_global_has_random_seed(self) -> None:
        """Test global contient random_seed."""
        defaults = get_default_hyperparameters()
        global_config = defaults["global"]

        assert isinstance(global_config, dict)
        assert "random_seed" in global_config
        assert global_config["random_seed"] == 42
