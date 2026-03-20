"""Tests Parsing Schemas - ISO 29119.

Document ID: ALICE-TEST-SCHEMAS-PARSING
Version: 1.0.0
Tests: 10

Classes:
- TestEchiquiersRawSchema: Tests for raw echiquiers validation (5 tests)
- TestJoueursRawSchema: Tests for raw joueurs validation (5 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from schemas.parsing_schemas import EchiquiersRawSchema, JoueursRawSchema


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


class TestEchiquiersRawSchema:
    """Tests for raw echiquiers DataFrame validation (ISO 5259)."""

    def test_valid_dataframe_passes(self, valid_echiquiers_df):
        EchiquiersRawSchema.validate(valid_echiquiers_df, lazy=True)

    def test_invalid_saison_raises(self, valid_echiquiers_df):
        valid_echiquiers_df["saison"] = [1900]
        with pytest.raises(SchemaErrors):
            EchiquiersRawSchema.validate(valid_echiquiers_df, lazy=True)

    def test_invalid_elo_raises(self, valid_echiquiers_df):
        valid_echiquiers_df["blanc_elo"] = [5000]
        with pytest.raises(SchemaErrors):
            EchiquiersRawSchema.validate(valid_echiquiers_df, lazy=True)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"saison": [2025]})
        with pytest.raises(SchemaErrors):
            EchiquiersRawSchema.validate(df, lazy=True)

    def test_resultat_forfait_2_passes(self, valid_echiquiers_df):
        """FFE forfait score 2.0 is valid."""
        valid_echiquiers_df["resultat_blanc"] = [2.0]
        valid_echiquiers_df["resultat_noir"] = [0.0]
        EchiquiersRawSchema.validate(valid_echiquiers_df, lazy=True)

    def test_echiquier_16_passes(self, valid_echiquiers_df):
        """Top 16 has 16 boards."""
        valid_echiquiers_df["echiquier"] = [16]
        EchiquiersRawSchema.validate(valid_echiquiers_df, lazy=True)

    def test_diff_elo_coherence(self, valid_echiquiers_df):
        valid_echiquiers_df["diff_elo"] = [999]
        with pytest.raises(SchemaErrors):
            EchiquiersRawSchema.validate(valid_echiquiers_df, lazy=True)


class TestJoueursRawSchema:
    """Tests for raw joueurs DataFrame validation (ISO 5259)."""

    def test_valid_dataframe_passes(self, valid_joueurs_df):
        JoueursRawSchema.validate(valid_joueurs_df, lazy=True)

    def test_invalid_elo_type_raises(self, valid_joueurs_df):
        valid_joueurs_df["elo_type"] = ["X"]
        with pytest.raises(SchemaErrors):
            JoueursRawSchema.validate(valid_joueurs_df, lazy=True)

    def test_invalid_nr_ffe_pattern_raises(self, valid_joueurs_df):
        valid_joueurs_df["nr_ffe"] = ["12345"]
        with pytest.raises(SchemaErrors):
            JoueursRawSchema.validate(valid_joueurs_df, lazy=True)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"nom": ["DUPONT"]})
        with pytest.raises(SchemaErrors):
            JoueursRawSchema.validate(df, lazy=True)

    def test_elo_out_of_range_raises(self, valid_joueurs_df):
        valid_joueurs_df["elo"] = [4000]
        with pytest.raises(SchemaErrors):
            JoueursRawSchema.validate(valid_joueurs_df, lazy=True)
