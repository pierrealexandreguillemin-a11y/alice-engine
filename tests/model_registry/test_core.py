"""Tests Core - ISO 29119.

Document ID: ALICE-TEST-MODEL-CORE
Version: 1.0.0
Tests: 6 classes

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

import pandas as pd
import pytest

from scripts.model_registry import (
    DataLineage,
    EnvironmentInfo,
    compute_data_lineage,
    compute_dataframe_hash,
    compute_file_checksum,
    get_environment_info,
    get_git_info,
    get_package_versions,
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
