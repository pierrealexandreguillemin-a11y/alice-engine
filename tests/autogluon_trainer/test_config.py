"""Tests AutoGluon Config - ISO 42001.

Document ID: ALICE-TEST-AUTOGLUON-TRAINER-CONFIG
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from pathlib import Path

from scripts.autogluon.config import AutoGluonConfig, load_autogluon_config


class TestAutoGluonConfig:
    """Tests pour AutoGluonConfig."""

    def test_default_config(self) -> None:
        """Test configuration par defaut."""
        config = AutoGluonConfig()

        assert config.presets == "extreme"
        assert config.time_limit == 21600  # 6 hours default
        assert config.eval_metric == "roc_auc"
        assert config.num_bag_folds == 5
        assert config.num_stack_levels == 2
        assert config.random_seed == 42
        assert config.verbosity == 2

    def test_custom_config(self) -> None:
        """Test configuration personnalisee."""
        config = AutoGluonConfig(
            presets="high_quality",
            time_limit=1800,
            eval_metric="accuracy",
        )

        assert config.presets == "high_quality"
        assert config.time_limit == 1800
        assert config.eval_metric == "accuracy"

    def test_models_include_default(self) -> None:
        """Test modeles inclus par defaut."""
        config = AutoGluonConfig()

        assert "TabPFN" in config.models_include
        assert "CatBoost" in config.models_include
        assert "XGBoost" in config.models_include
        assert "LightGBM" in config.models_include

    def test_to_fit_kwargs(self) -> None:
        """Test conversion en kwargs pour fit()."""
        config = AutoGluonConfig(
            presets="best_quality",
            time_limit=600,
            num_bag_folds=3,
        )

        kwargs = config.to_fit_kwargs()

        assert kwargs["presets"] == "best_quality"
        assert kwargs["time_limit"] == 600
        assert kwargs["num_bag_folds"] == 3
        assert "eval_metric" not in kwargs

    def test_output_dir_default(self) -> None:
        """Test repertoire de sortie par defaut."""
        config = AutoGluonConfig()

        assert config.output_dir == Path("models/autogluon")


class TestLoadAutogluonConfig:
    """Tests pour load_autogluon_config."""

    def test_load_from_file(self, config_file: Path) -> None:
        """Test chargement depuis fichier."""
        config = load_autogluon_config(config_file)

        assert config.presets == "best_quality"
        assert config.time_limit == 60
        assert config.num_bag_folds == 3
        assert config.tabpfn_enabled is True
        assert config.tabpfn_n_ensemble == 8

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Test fichier manquant retourne defauts."""
        config = load_autogluon_config(tmp_path / "nonexistent.yaml")

        assert config.presets == "extreme"
        assert config.time_limit == 21600  # 6 hours default

    def test_load_empty_autogluon_section(self, tmp_path: Path) -> None:
        """Test section autogluon vide."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("other_section: true", encoding="utf-8")

        config = load_autogluon_config(config_path)

        assert config.presets == "extreme"
