# tests/test_model_registry.py
"""Tests pour model_registry.py - ISO 29119.

Tests unitaires pour la normalisation des modèles production.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.model_registry import (
    DataLineage,
    EnvironmentInfo,
    ModelArtifact,
    ProductionModelCard,
    apply_retention_policy,
    compute_data_lineage,
    compute_dataframe_hash,
    compute_file_checksum,
    compute_model_signature,
    extract_feature_importance,
    generate_signing_key,
    get_environment_info,
    get_git_info,
    get_package_versions,
    get_retention_status,
    list_model_versions,
    rollback_to_version,
    validate_dataframe_schema,
    validate_model_integrity,
    validate_train_valid_test_schema,
    verify_model_signature,
)


class TestComputeFileChecksum:
    """Tests pour compute_file_checksum."""

    def test_checksum_consistent(self, tmp_path: Path) -> None:
        """Test checksum est cohérent pour même fichier."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        checksum1 = compute_file_checksum(test_file)
        checksum2 = compute_file_checksum(test_file)

        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA-256 hex

    def test_checksum_different_content(self, tmp_path: Path) -> None:
        """Test checksums différents pour contenus différents."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content A")
        file2.write_text("Content B")

        checksum1 = compute_file_checksum(file1)
        checksum2 = compute_file_checksum(file2)

        assert checksum1 != checksum2


class TestComputeDataframeHash:
    """Tests pour compute_dataframe_hash."""

    def test_hash_consistent(self) -> None:
        """Test hash cohérent pour même DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        hash1 = compute_dataframe_hash(df)
        hash2 = compute_dataframe_hash(df)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_hash_different_data(self) -> None:
        """Test hash différent pour données différentes."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})

        hash1 = compute_dataframe_hash(df1)
        hash2 = compute_dataframe_hash(df2)

        assert hash1 != hash2


class TestGetGitInfo:
    """Tests pour get_git_info."""

    def test_returns_tuple(self) -> None:
        """Test retourne un tuple de 3 éléments."""
        result = get_git_info()

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_commit_hash_format(self) -> None:
        """Test format du commit hash si présent."""
        commit, branch, dirty = get_git_info()

        if commit is not None:
            assert len(commit) == 40  # Full SHA
            assert all(c in "0123456789abcdef" for c in commit)


class TestGetPackageVersions:
    """Tests pour get_package_versions."""

    def test_returns_dict(self) -> None:
        """Test retourne un dictionnaire."""
        versions = get_package_versions()

        assert isinstance(versions, dict)

    def test_contains_common_packages(self) -> None:
        """Test contient les packages courants."""
        versions = get_package_versions()

        # Au moins numpy et pandas devraient être présents
        assert "numpy" in versions or "pandas" in versions


class TestGetEnvironmentInfo:
    """Tests pour get_environment_info."""

    def test_returns_environment_info(self) -> None:
        """Test retourne EnvironmentInfo."""
        env = get_environment_info()

        assert isinstance(env, EnvironmentInfo)
        assert env.python_version is not None
        assert env.platform_system is not None

    def test_to_dict(self) -> None:
        """Test conversion en dictionnaire."""
        env = get_environment_info()
        env_dict = env.to_dict()

        assert "python_version" in env_dict
        assert "platform" in env_dict
        assert "git" in env_dict
        assert "packages" in env_dict


class TestComputeDataLineage:
    """Tests pour compute_data_lineage."""

    @pytest.fixture
    def sample_data(self, tmp_path: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Crée des données de test."""
        train = pd.DataFrame({"feature": [1, 2, 3], "resultat_blanc": [1.0, 0.0, 1.0]})
        valid = pd.DataFrame({"feature": [4, 5], "resultat_blanc": [0.0, 1.0]})
        test = pd.DataFrame({"feature": [6, 7], "resultat_blanc": [1.0, 0.0]})

        train_path = tmp_path / "train.parquet"
        valid_path = tmp_path / "valid.parquet"
        test_path = tmp_path / "test.parquet"

        train.to_parquet(train_path)
        valid.to_parquet(valid_path)
        test.to_parquet(test_path)

        return tmp_path, train, valid, test

    def test_compute_lineage(self, sample_data: tuple) -> None:
        """Test calcul de la lineage."""
        data_dir, train, valid, test = sample_data

        lineage = compute_data_lineage(
            data_dir / "train.parquet",
            data_dir / "valid.parquet",
            data_dir / "test.parquet",
            train,
            valid,
            test,
        )

        assert isinstance(lineage, DataLineage)
        assert lineage.train_samples == 3
        assert lineage.valid_samples == 2
        assert lineage.test_samples == 2

    def test_lineage_to_dict(self, sample_data: tuple) -> None:
        """Test conversion lineage en dict."""
        data_dir, train, valid, test = sample_data

        lineage = compute_data_lineage(
            data_dir / "train.parquet",
            data_dir / "valid.parquet",
            data_dir / "test.parquet",
            train,
            valid,
            test,
        )
        lineage_dict = lineage.to_dict()

        assert "train" in lineage_dict
        assert "valid" in lineage_dict
        assert "test" in lineage_dict
        assert lineage_dict["train"]["samples"] == 3


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


class TestValidateModelIntegrity:
    """Tests pour validate_model_integrity (ISO 27001)."""

    def test_valid_integrity(self, tmp_path: Path) -> None:
        """Test validation réussie avec checksum correct."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")
        checksum = compute_file_checksum(model_file)

        artifact = ModelArtifact(
            name="CatBoost",
            path=model_file,
            format=".cbm",
            checksum=checksum,
            size_bytes=model_file.stat().st_size,
        )

        result = validate_model_integrity(artifact)

        assert result is True

    def test_invalid_checksum(self, tmp_path: Path) -> None:
        """Test validation échouée avec checksum incorrect."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")

        artifact = ModelArtifact(
            name="CatBoost",
            path=model_file,
            format=".cbm",
            checksum="invalid_checksum_" + "0" * 48,  # Wrong checksum
            size_bytes=model_file.stat().st_size,
        )

        result = validate_model_integrity(artifact)

        assert result is False

    def test_missing_file(self, tmp_path: Path) -> None:
        """Test validation échouée avec fichier manquant."""
        artifact = ModelArtifact(
            name="CatBoost",
            path=tmp_path / "nonexistent.cbm",
            format=".cbm",
            checksum="a" * 64,
            size_bytes=100,
        )

        result = validate_model_integrity(artifact)

        assert result is False


class TestExtractFeatureImportance:
    """Tests pour extract_feature_importance (ISO 42001)."""

    def test_catboost_feature_importance(self) -> None:
        """Test extraction importance CatBoost."""
        mock_model = MagicMock()
        mock_model.get_feature_importance.return_value = np.array([0.3, 0.5, 0.2])

        feature_names = ["elo", "niveau", "ronde"]
        importance = extract_feature_importance(mock_model, "CatBoost", feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == 3
        # Normalisé et trié par importance décroissante
        assert list(importance.keys())[0] == "niveau"  # 0.5 is highest
        assert abs(sum(importance.values()) - 1.0) < 0.001  # Sum to 1

    def test_xgboost_feature_importance(self) -> None:
        """Test extraction importance XGBoost."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.4, 0.4, 0.2])

        feature_names = ["elo", "niveau", "ronde"]
        importance = extract_feature_importance(mock_model, "XGBoost", feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert abs(sum(importance.values()) - 1.0) < 0.001

    def test_lightgbm_feature_importance(self) -> None:
        """Test extraction importance LightGBM."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.1, 0.6, 0.3])

        feature_names = ["elo", "niveau", "ronde"]
        importance = extract_feature_importance(mock_model, "LightGBM", feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert list(importance.keys())[0] == "niveau"  # 0.6 is highest

    def test_unknown_model_returns_empty(self) -> None:
        """Test modèle inconnu retourne dict vide."""
        mock_model = MagicMock(spec=[])  # No feature importance methods

        feature_names = ["elo", "niveau"]
        importance = extract_feature_importance(mock_model, "UnknownModel", feature_names)

        assert importance == {}

    def test_empty_features_list(self) -> None:
        """Test liste features vide."""
        mock_model = MagicMock()
        mock_model.get_feature_importance.return_value = np.array([])

        importance = extract_feature_importance(mock_model, "CatBoost", [])

        assert importance == {}


# ==============================================================================
# P2 TESTS - Signature HMAC, Schema validation, Retention policy
# ==============================================================================


class TestModelSignature:
    """Tests pour signature HMAC-SHA256 (ISO 27001)."""

    def test_generate_signing_key(self) -> None:
        """Test génération clé de signature."""
        key = generate_signing_key()

        assert len(key) == 64  # 32 bytes hex
        assert all(c in "0123456789abcdef" for c in key)

    def test_compute_signature(self, tmp_path: Path) -> None:
        """Test calcul signature."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")
        key = generate_signing_key()

        signature = compute_model_signature(model_file, key)

        assert len(signature) == 64  # SHA-256 hex

    def test_signature_deterministic(self, tmp_path: Path) -> None:
        """Test signature déterministe pour même fichier et clé."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")
        key = generate_signing_key()

        sig1 = compute_model_signature(model_file, key)
        sig2 = compute_model_signature(model_file, key)

        assert sig1 == sig2

    def test_different_key_different_signature(self, tmp_path: Path) -> None:
        """Test clés différentes produisent signatures différentes."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")
        key1 = generate_signing_key()
        key2 = generate_signing_key()

        sig1 = compute_model_signature(model_file, key1)
        sig2 = compute_model_signature(model_file, key2)

        assert sig1 != sig2

    def test_verify_valid_signature(self, tmp_path: Path) -> None:
        """Test vérification signature valide."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")
        key = generate_signing_key()
        signature = compute_model_signature(model_file, key)

        result = verify_model_signature(model_file, signature, key)

        assert result is True

    def test_verify_invalid_signature(self, tmp_path: Path) -> None:
        """Test vérification signature invalide."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")
        key = generate_signing_key()

        result = verify_model_signature(model_file, "invalid" + "0" * 56, key)

        assert result is False

    def test_verify_missing_file(self, tmp_path: Path) -> None:
        """Test vérification fichier manquant."""
        key = generate_signing_key()

        result = verify_model_signature(tmp_path / "nonexistent.cbm", "a" * 64, key)

        assert result is False


class TestSchemaValidation:
    """Tests pour validation schema DataFrame (ISO 5259)."""

    def test_valid_schema(self) -> None:
        """Test schema valide."""
        df = pd.DataFrame(
            {
                "resultat_blanc": [1.0, 0.0, 1.0],
                "blanc_elo": [1500, 1600, 1700],
                "noir_elo": [1450, 1550, 1650],
                "diff_elo": [50, 50, 50],
            }
        )

        result = validate_dataframe_schema(df)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_required_column(self) -> None:
        """Test colonne requise manquante."""
        df = pd.DataFrame(
            {
                "blanc_elo": [1500, 1600],
                "noir_elo": [1450, 1550],
            }
        )

        result = validate_dataframe_schema(df)

        assert result.is_valid is False
        assert any("resultat_blanc" in e for e in result.errors)

    def test_non_numeric_column(self) -> None:
        """Test colonne non-numérique."""
        df = pd.DataFrame(
            {
                "resultat_blanc": [1.0, 0.0],
                "blanc_elo": ["high", "low"],  # Should be numeric
                "noir_elo": [1450, 1550],
                "diff_elo": [50, 50],
            }
        )

        result = validate_dataframe_schema(df)

        assert result.is_valid is False
        assert any("blanc_elo" in e and "numeric" in e for e in result.errors)

    def test_empty_dataframe(self) -> None:
        """Test DataFrame vide."""
        df = pd.DataFrame({"resultat_blanc": []})

        result = validate_dataframe_schema(df)

        assert result.is_valid is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_high_null_ratio_warning(self) -> None:
        """Test warning pour taux de null élevé."""
        df = pd.DataFrame(
            {
                "resultat_blanc": [1.0, None, None, None, None],  # 80% null
                "blanc_elo": [1500, 1600, 1700, 1800, 1900],
            }
        )

        result = validate_dataframe_schema(df, required_columns=set())

        assert len(result.warnings) > 0
        assert any("null" in w.lower() for w in result.warnings)

    def test_allow_missing_columns(self) -> None:
        """Test allow_missing=True."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        result = validate_dataframe_schema(df, allow_missing=True)

        assert result.is_valid is True
        assert len(result.warnings) > 0  # Warning instead of error

    def test_train_valid_test_consistency(self) -> None:
        """Test cohérence train/valid/test."""
        train = pd.DataFrame(
            {
                "resultat_blanc": [1.0] * 100,
                "blanc_elo": [1500] * 100,
                "noir_elo": [1400] * 100,
                "diff_elo": [100] * 100,
            }
        )
        valid = train.copy()
        test = train.head(20).copy()

        result = validate_train_valid_test_schema(train, valid, test)

        assert result.is_valid is True


class TestRetentionPolicy:
    """Tests pour politique de rétention (ISO 27001)."""

    def test_retention_under_limit(self, tmp_path: Path) -> None:
        """Test rétention sous la limite."""
        # Créer 3 versions (sous DEFAULT_MAX_VERSIONS=10)
        for i in range(3):
            v = tmp_path / f"v2024010{i}_120000"
            v.mkdir()
            (v / "metadata.json").write_text("{}")

        deleted = apply_retention_policy(tmp_path)

        assert len(deleted) == 0
        assert len(list_model_versions(tmp_path)) == 3

    def test_retention_over_limit(self, tmp_path: Path) -> None:
        """Test rétention au-dessus de la limite."""
        # Créer 5 versions avec max_versions=3
        for i in range(5):
            v = tmp_path / f"v2024010{i}_120000"
            v.mkdir()
            (v / "metadata.json").write_text("{}")

        deleted = apply_retention_policy(tmp_path, max_versions=3)

        assert len(deleted) == 2  # 5 - 3 = 2
        assert len(list_model_versions(tmp_path)) == 3

    def test_retention_dry_run(self, tmp_path: Path) -> None:
        """Test dry_run ne supprime pas."""
        for i in range(5):
            v = tmp_path / f"v2024010{i}_120000"
            v.mkdir()
            (v / "metadata.json").write_text("{}")

        deleted = apply_retention_policy(tmp_path, max_versions=3, dry_run=True)

        assert len(deleted) == 2
        assert len(list_model_versions(tmp_path)) == 5  # Rien supprimé

    def test_retention_keeps_newest(self, tmp_path: Path) -> None:
        """Test garde les plus récentes."""
        versions = ["v20240101_120000", "v20240102_120000", "v20240103_120000"]
        for v_name in versions:
            v = tmp_path / v_name
            v.mkdir()
            (v / "metadata.json").write_text("{}")

        apply_retention_policy(tmp_path, max_versions=2)

        remaining = list_model_versions(tmp_path)
        assert len(remaining) == 2
        # Les plus récentes gardées
        assert remaining[0].name == "v20240103_120000"
        assert remaining[1].name == "v20240102_120000"

    def test_get_retention_status(self, tmp_path: Path) -> None:
        """Test statut rétention."""
        for i in range(3):
            v = tmp_path / f"v2024010{i}_120000"
            v.mkdir()
            (v / "metadata.json").write_text("{}")

        status = get_retention_status(tmp_path, max_versions=5)

        assert status["current_count"] == 3
        assert status["max_versions"] == 5
        assert status["versions_to_delete"] == 0
        assert status["retention_applied"] is True

    def test_retention_status_over_limit(self, tmp_path: Path) -> None:
        """Test statut quand au-dessus limite."""
        for i in range(5):
            v = tmp_path / f"v2024010{i}_120000"
            v.mkdir()
            (v / "metadata.json").write_text("{}")

        status = get_retention_status(tmp_path, max_versions=3)

        assert status["current_count"] == 5
        assert status["versions_to_delete"] == 2
        assert status["retention_applied"] is False
