"""Tests Artifacts - ISO 29119.

Document ID: ALICE-TEST-MODEL-ARTIFACTS
Version: 1.0.0
Tests: 4 classes

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

from scripts.model_registry import (
    DataLineage,
    EnvironmentInfo,
    ModelArtifact,
    ProductionModelCard,
    list_model_versions,
    rollback_to_version,
)


class TestModelArtifact:
    """Tests pour ModelArtifact dataclass."""

    def test_artifact_creation(self, tmp_path: Path) -> None:
        """Test création d'artefact."""
        test_file = tmp_path / "model.cbm"
        test_file.write_text("model data")

        artifact = ModelArtifact(
            name="CatBoost",
            path=test_file,
            format=".cbm",
            checksum="abc123" * 10 + "abcd",
            size_bytes=100,
        )

        assert artifact.name == "CatBoost"
        assert artifact.format == ".cbm"

    def test_artifact_to_dict(self, tmp_path: Path) -> None:
        """Test conversion artefact en dict."""
        artifact = ModelArtifact(
            name="XGBoost",
            path=tmp_path / "xgboost.ubj",
            format=".ubj",
            checksum="def456" * 10 + "defg",
            size_bytes=200,
        )
        artifact_dict = artifact.to_dict()

        assert artifact_dict["name"] == "XGBoost"
        assert artifact_dict["format"] == ".ubj"
        assert artifact_dict["size_bytes"] == 200


class TestProductionModelCard:
    """Tests pour ProductionModelCard dataclass."""

    def test_model_card_to_dict(self) -> None:
        """Test conversion Model Card en dict."""
        env = EnvironmentInfo(
            python_version="3.11",
            platform_system="Linux",
            platform_release="5.0",
            platform_machine="x86_64",
            git_commit="abc123",
            git_branch="main",
            git_dirty=False,
            packages={"numpy": "1.25.0"},
        )
        lineage = DataLineage(
            train_path="/data/train.parquet",
            valid_path="/data/valid.parquet",
            test_path="/data/test.parquet",
            train_samples=1000,
            valid_samples=100,
            test_samples=200,
            train_hash="hash1",
            valid_hash="hash2",
            test_hash="hash3",
            feature_count=10,
            target_distribution={"positive_ratio": 0.5},
            created_at="2024-01-01T00:00:00",
        )

        card = ProductionModelCard(
            version="v20240101",
            created_at="2024-01-01T00:00:00",
            environment=env,
            data_lineage=lineage,
            artifacts=[],
            metrics={"CatBoost": {"auc_roc": 0.85}},
            feature_importance={"CatBoost": {"elo": 0.3}},
            hyperparameters={"catboost": {"iterations": 1000}},
            best_model={"name": "CatBoost", "auc": 0.85},
        )

        card_dict = card.to_dict()

        assert card_dict["version"] == "v20240101"
        assert "environment" in card_dict
        assert "data_lineage" in card_dict
        assert "metrics" in card_dict


class TestListModelVersions:
    """Tests pour list_model_versions."""

    def test_list_empty_dir(self, tmp_path: Path) -> None:
        """Test liste répertoire vide."""
        versions = list_model_versions(tmp_path)

        assert versions == []

    def test_list_versions(self, tmp_path: Path) -> None:
        """Test liste les versions avec metadata."""
        # Créer des versions
        v1 = tmp_path / "v20240101_120000"
        v2 = tmp_path / "v20240102_120000"
        v1.mkdir()
        v2.mkdir()

        # Ajouter metadata.json
        (v1 / "metadata.json").write_text("{}")
        (v2 / "metadata.json").write_text("{}")

        versions = list_model_versions(tmp_path)

        assert len(versions) == 2
        # Plus récent en premier
        assert versions[0].name == "v20240102_120000"


class TestRollback:
    """Tests pour rollback_to_version."""

    def test_rollback_nonexistent_version(self, tmp_path: Path) -> None:
        """Test rollback version inexistante."""
        result = rollback_to_version(tmp_path, "v_nonexistent")

        assert result is False

    def test_rollback_existing_version(self, tmp_path: Path) -> None:
        """Test rollback version existante."""
        version_dir = tmp_path / "v20240101"
        version_dir.mkdir()
        (version_dir / "metadata.json").write_text("{}")

        result = rollback_to_version(tmp_path, "v20240101")

        assert result is True


# ==============================================================================
# P1 TESTS - Validation intégrité, Feature importance, Intégration
# ==============================================================================
