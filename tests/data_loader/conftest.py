"""Fixtures DataLoader Service - ISO 29119.

Document ID: ALICE-TEST-DATA-LOADER-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

import pandas as pd
import pytest

from services.data_loader import DataLoader


@pytest.fixture
def loader() -> DataLoader:
    """Fixture loader sans configuration."""
    return DataLoader()


@pytest.fixture
def loader_with_mongodb() -> DataLoader:
    """Fixture loader avec URI MongoDB."""
    return DataLoader(mongodb_uri="mongodb://localhost:27017/alice")


@pytest.fixture
def loader_with_dataset(tmp_path: Path) -> DataLoader:
    """Fixture loader avec chemin dataset."""
    return DataLoader(dataset_path=tmp_path)


@pytest.fixture
def sample_parquet_data(tmp_path: Path) -> Path:
    """Fixture créant un fichier parquet de test."""
    df = pd.DataFrame(
        [
            {"saison": 2023, "blanc_nom": "A", "noir_nom": "B", "resultat_blanc": 1.0},
            {"saison": 2024, "blanc_nom": "C", "noir_nom": "D", "resultat_blanc": 0.5},
            {"saison": 2024, "blanc_nom": "E", "noir_nom": "F", "resultat_blanc": 0.0},
            {"saison": 2025, "blanc_nom": "G", "noir_nom": "H", "resultat_blanc": 1.0},
        ]
    )
    parquet_path = tmp_path / "echiquiers.parquet"
    df.to_parquet(parquet_path)
    return tmp_path
