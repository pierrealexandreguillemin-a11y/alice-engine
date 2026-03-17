"""Tests Parsing Validation - ISO 29119.

Document ID: ALICE-TEST-SCHEMAS-PARSING-VALID
Version: 1.0.0
Tests: 6

Classes:
- TestValidateRawEchiquiers: Tests validate_raw_echiquiers (3 tests)
- TestValidateRawJoueurs: Tests validate_raw_joueurs (3 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
"""

from __future__ import annotations

import pandas as pd
import pytest

from schemas.parsing_validation import validate_raw_echiquiers, validate_raw_joueurs


@pytest.fixture()
def valid_echiquiers_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "saison": [2025],
            "competition": ["Interclubs"],
            "division": ["Nationale_1"],
            "groupe": ["Groupe_A"],
            "ronde": [1],
            "echiquier": [1],
            "blanc_nom": ["DUPONT Jean"],
            "noir_nom": ["MARTIN Paul"],
            "blanc_elo": [2100],
            "noir_elo": [1950],
            "resultat_blanc": [1.0],
            "resultat_noir": [0.0],
            "type_resultat": ["victoire_blanc"],
            "diff_elo": [150],
        }
    )


@pytest.fixture()
def valid_joueurs_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nr_ffe": ["K59857"],
            "id_ffe": [672495],
            "nom": ["DUPONT"],
            "prenom": ["Jean"],
            "nom_complet": ["DUPONT Jean"],
            "elo": [1567],
            "elo_type": ["F"],
            "categorie": ["SenM"],
            "club": ["Echiquier de Bigorre"],
        }
    )


class TestValidateRawEchiquiers:
    """Tests for validate_raw_echiquiers (ISO 29119)."""

    def test_valid_returns_report_no_errors(self, valid_echiquiers_df):
        report = validate_raw_echiquiers(valid_echiquiers_df)
        assert report.is_valid is True
        assert report.errors == []

    def test_report_persisted_to_disk(self, valid_echiquiers_df, tmp_path):
        report = validate_raw_echiquiers(valid_echiquiers_df, report_dir=tmp_path)
        assert (tmp_path / "raw_echiquiers_report.json").exists()

    def test_invalid_data_returns_errors(self):
        df = pd.DataFrame(
            {
                "saison": [1900],
                "competition": ["X"],
                "division": ["X"],
                "groupe": ["X"],
                "ronde": [1],
                "echiquier": [1],
                "blanc_nom": ["A"],
                "noir_nom": ["B"],
                "blanc_elo": [5000],
                "noir_elo": [1000],
                "resultat_blanc": [1.0],
                "resultat_noir": [0.0],
                "type_resultat": ["victoire_blanc"],
                "diff_elo": [4000],
            }
        )
        report = validate_raw_echiquiers(df)
        assert report.is_valid is False
        assert len(report.errors) > 0


class TestValidateRawJoueurs:
    """Tests for validate_raw_joueurs (ISO 29119)."""

    def test_valid_returns_report_no_errors(self, valid_joueurs_df):
        report = validate_raw_joueurs(valid_joueurs_df)
        assert report.is_valid is True
        assert report.errors == []

    def test_report_persisted_to_disk(self, valid_joueurs_df, tmp_path):
        report = validate_raw_joueurs(valid_joueurs_df, report_dir=tmp_path)
        assert (tmp_path / "raw_joueurs_report.json").exists()

    def test_invalid_data_returns_errors(self):
        df = pd.DataFrame(
            {
                "nr_ffe": ["12345"],
                "id_ffe": [1],
                "nom": ["X"],
                "prenom": ["Y"],
                "nom_complet": ["X Y"],
                "elo": [5000],
                "elo_type": ["Z"],
                "categorie": ["SenM"],
                "club": ["Club"],
            }
        )
        report = validate_raw_joueurs(df)
        assert report.is_valid is False
        assert len(report.errors) > 0
