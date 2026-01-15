"""Tests Calculate Presence Features - ISO 29119.

Document ID: ALICE-TEST-FEATURES-ALI-PRESENCE
Version: 1.0.0

Tests pour calculate_presence_features.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd
import pytest

from scripts.features.ali import calculate_presence_features


class TestCalculatePresenceFeatures:
    """Tests pour calculate_presence_features()."""

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test avec DataFrame vide retourne DataFrame vide."""
        result = calculate_presence_features(empty_df)
        assert result.empty

    def test_returns_dataframe(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test retourne un DataFrame."""
        result = calculate_presence_features(sample_echiquiers)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test colonnes requises présentes."""
        result = calculate_presence_features(sample_echiquiers)

        required_cols = [
            "joueur_nom",
            "saison",
            "taux_presence_saison",
            "derniere_presence",
            "nb_rondes_jouees",
            "nb_rondes_total",
            "regularite",
        ]
        for col in required_cols:
            assert col in result.columns, f"Colonne manquante: {col}"

    def test_regularite_regulier_threshold(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test seuil régularité 'regulier' (>70%)."""
        result = calculate_presence_features(sample_echiquiers)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        assert len(joueur_a) == 1
        assert joueur_a.iloc[0]["regularite"] == "regulier"

    def test_regularite_rare_threshold(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test seuil régularité 'rare' (<30%)."""
        result = calculate_presence_features(sample_echiquiers)

        joueur_c = result[result["joueur_nom"] == "Joueur C"]
        assert len(joueur_c) == 1
        assert joueur_c.iloc[0]["regularite"] == "rare"

    def test_regularite_occasionnel_threshold(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test seuil régularité 'occasionnel' (30-70%)."""
        result = calculate_presence_features(sample_echiquiers)

        joueur_b = result[result["joueur_nom"] == "Joueur B"]
        assert len(joueur_b) == 1
        assert joueur_b.iloc[0]["regularite"] == "occasionnel"

    def test_taux_presence_calculation(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test calcul taux de présence correct."""
        result = calculate_presence_features(sample_echiquiers)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        assert joueur_a.iloc[0]["nb_rondes_jouees"] == 7
        assert joueur_a.iloc[0]["taux_presence_saison"] == pytest.approx(7 / 9, rel=0.01)

    def test_derniere_presence_calculation(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test calcul dernière présence."""
        result = calculate_presence_features(sample_echiquiers)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        assert joueur_a.iloc[0]["derniere_presence"] == 2

    def test_filter_by_saison(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test filtrage par saison."""
        result = calculate_presence_features(sample_echiquiers, saison=2025)
        assert len(result) > 0
        assert all(result["saison"] == 2025)

    def test_filter_nonexistent_saison(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test filtrage saison inexistante retourne vide."""
        result = calculate_presence_features(sample_echiquiers, saison=2020)
        assert result.empty

    def test_deduplication_blanc_noir(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test déduplication joueurs jouant blanc et noir."""
        result = calculate_presence_features(sample_echiquiers)
        assert result.groupby(["joueur_nom", "saison"]).size().max() == 1
