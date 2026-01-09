"""Module: test_features_recent_form.py - Tests Recent Form Features.

Tests unitaires pour le module recent_form.py - forme recente joueur.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing (unit tests, boundary, edge cases)
- ISO/IEC 5259:2024 - Data Quality for ML (feature validation)
- ISO/IEC 25010:2023 - System Quality (testability)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

import pandas as pd
import pytest

from scripts.features.recent_form import _calculate_tendance, calculate_recent_form

# ==============================================================================
# FIXTURES (ISO 29119-3: Test Design)
# ==============================================================================


@pytest.fixture
def sample_matches_for_form() -> pd.DataFrame:
    """Fixture avec matchs pour tester le calcul de forme recente."""
    return pd.DataFrame(
        [
            # Joueur A - 6 matchs avec blanc (forme croissante)
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Adversaire 1",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
                "date": "2025-01-01",
            },
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Adversaire 2",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
                "date": "2025-01-08",
            },
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Adversaire 3",
                "resultat_blanc": 0.5,
                "resultat_noir": 0.5,
                "type_resultat": "nulle",
                "date": "2025-01-15",
            },
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Adversaire 4",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-22",
            },
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Adversaire 5",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-29",
            },
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Adversaire 6",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-02-05",
            },
            # Joueur B - 5 matchs avec noir (forme decroissante)
            {
                "blanc_nom": "Adversaire 7",
                "noir_nom": "Joueur B",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
                "date": "2025-01-01",
            },
            {
                "blanc_nom": "Adversaire 8",
                "noir_nom": "Joueur B",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
                "date": "2025-01-08",
            },
            {
                "blanc_nom": "Adversaire 9",
                "noir_nom": "Joueur B",
                "resultat_blanc": 0.5,
                "resultat_noir": 0.5,
                "type_resultat": "nulle",
                "date": "2025-01-15",
            },
            {
                "blanc_nom": "Adversaire 10",
                "noir_nom": "Joueur B",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-22",
            },
            {
                "blanc_nom": "Adversaire 11",
                "noir_nom": "Joueur B",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-29",
            },
            # Joueur C - seulement 3 matchs (pas assez pour window=5)
            {
                "blanc_nom": "Joueur C",
                "noir_nom": "Adversaire 12",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-01",
            },
            {
                "blanc_nom": "Joueur C",
                "noir_nom": "Adversaire 13",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-08",
            },
            {
                "blanc_nom": "Joueur C",
                "noir_nom": "Adversaire 14",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-15",
            },
        ]
    )


@pytest.fixture
def matches_with_forfaits() -> pd.DataFrame:
    """Fixture avec forfaits pour tester filtrage."""
    return pd.DataFrame(
        [
            # Matchs joues
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 1",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-01",
            },
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 2",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-08",
            },
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 3",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-15",
            },
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 4",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-22",
            },
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 5",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-29",
            },
            # Forfaits (doivent etre ignores)
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 6",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "forfait_noir",
                "date": "2025-02-05",
            },
            {
                "blanc_nom": "Adversaire 7",
                "noir_nom": "Joueur D",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "forfait_blanc",
                "date": "2025-02-12",
            },
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 8",
                "resultat_blanc": 0.0,
                "resultat_noir": 0.0,
                "type_resultat": "non_joue",
                "date": "2025-02-19",
            },
        ]
    )


# ==============================================================================
# TESTS: calculate_recent_form (ISO 29119-4: Test Techniques)
# ==============================================================================


class TestCalculateRecentForm:
    """Tests pour calculate_recent_form."""

    def test_basic_form_calculation(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test calcul forme basique avec donnees valides."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        assert not result.empty
        assert "joueur_nom" in result.columns
        assert "forme_recente" in result.columns
        assert "nb_matchs_forme" in result.columns
        assert "forme_tendance" in result.columns

    def test_form_values_in_valid_range(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test que forme_recente est dans [0, 1]."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        assert result["forme_recente"].min() >= 0.0
        assert result["forme_recente"].max() <= 1.0

    def test_joueur_a_hausse_tendance(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test Joueur A avec forme croissante = tendance hausse."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        assert len(joueur_a) == 1
        assert joueur_a.iloc[0]["forme_tendance"] == "hausse"

    def test_joueur_b_baisse_tendance(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test Joueur B avec forme decroissante = tendance baisse."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        joueur_b = result[result["joueur_nom"] == "Joueur B"]
        assert len(joueur_b) == 1
        assert joueur_b.iloc[0]["forme_tendance"] == "baisse"

    def test_joueur_excluded_if_insufficient_matches(
        self, sample_matches_for_form: pd.DataFrame
    ) -> None:
        """Test Joueur C avec <5 matchs est exclu."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        joueur_c = result[result["joueur_nom"] == "Joueur C"]
        assert len(joueur_c) == 0

    def test_empty_dataframe_returns_empty(self) -> None:
        """Test DataFrame vide retourne DataFrame vide."""
        result = calculate_recent_form(pd.DataFrame(), window=5)

        assert result.empty

    def test_forfaits_excluded_from_calculation(self, matches_with_forfaits: pd.DataFrame) -> None:
        """Test ISO 5259: forfaits exclus du calcul forme."""
        result = calculate_recent_form(matches_with_forfaits, window=5)

        # Joueur D a 5 matchs joues (exclut les 3 forfaits)
        joueur_d = result[result["joueur_nom"] == "Joueur D"]
        assert len(joueur_d) == 1
        # Forme parfaite car 5 victoires
        assert joueur_d.iloc[0]["forme_recente"] == 1.0

    def test_custom_window_size(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test avec taille de fenetre personnalisee."""
        result = calculate_recent_form(sample_matches_for_form, window=3)

        # Joueur C a maintenant assez de matchs (3)
        joueur_c = result[result["joueur_nom"] == "Joueur C"]
        assert len(joueur_c) == 1

    def test_missing_player_columns_returns_empty(self) -> None:
        """Test colonnes joueur manquantes retourne vide."""
        # DataFrame avec type_resultat mais sans colonnes blanc_nom/noir_nom
        df = pd.DataFrame(
            {
                "autre_colonne": [1, 2, 3],
                "type_resultat": ["victoire_blanc", "nulle", "victoire_noir"],
            }
        )
        result = calculate_recent_form(df, window=5)

        # Sans colonnes joueur, retourne DataFrame vide
        assert result.empty


# ==============================================================================
# TESTS: _calculate_tendance (ISO 29119-4: Boundary Value Analysis)
# ==============================================================================


class TestCalculateTendance:
    """Tests pour _calculate_tendance (fonction interne)."""

    def test_tendance_hausse(self) -> None:
        """Test detection tendance hausse (delta > 0.1)."""
        # Premiere moitie: 0, 0 = 0.0 moyenne
        # Seconde moitie: 1, 1 = 1.0 moyenne
        df = pd.DataFrame({"resultat": [0.0, 0.0, 1.0, 1.0]})
        result = _calculate_tendance(df, "resultat", 4)
        assert result == "hausse"

    def test_tendance_baisse(self) -> None:
        """Test detection tendance baisse (delta < -0.1)."""
        # Premiere moitie: 1, 1 = 1.0 moyenne
        # Seconde moitie: 0, 0 = 0.0 moyenne
        df = pd.DataFrame({"resultat": [1.0, 1.0, 0.0, 0.0]})
        result = _calculate_tendance(df, "resultat", 4)
        assert result == "baisse"

    def test_tendance_stable(self) -> None:
        """Test detection tendance stable (|delta| <= 0.1)."""
        # Premiere moitie: 0.5, 0.5 = 0.5 moyenne
        # Seconde moitie: 0.5, 0.5 = 0.5 moyenne
        df = pd.DataFrame({"resultat": [0.5, 0.5, 0.5, 0.5]})
        result = _calculate_tendance(df, "resultat", 4)
        assert result == "stable"

    def test_tendance_boundary_hausse(self) -> None:
        """Test limite hausse exactement a 0.1 (devrait etre stable)."""
        # Premiere moitie: 0.4, 0.5 = 0.45 moyenne
        # Seconde moitie: 0.5, 0.6 = 0.55 moyenne (delta = 0.1)
        df = pd.DataFrame({"resultat": [0.4, 0.5, 0.5, 0.6]})
        result = _calculate_tendance(df, "resultat", 4)
        # delta = 0.1 n'est pas > 0.1 donc stable
        assert result == "stable"

    def test_tendance_boundary_above_threshold(self) -> None:
        """Test juste au-dessus du seuil 0.1."""
        # Premiere moitie: 0, 0 = 0.0
        # Seconde moitie: 0.2, 0.2 = 0.2 (delta = 0.2 > 0.1)
        df = pd.DataFrame({"resultat": [0.0, 0.0, 0.2, 0.2]})
        result = _calculate_tendance(df, "resultat", 4)
        assert result == "hausse"

    def test_tendance_with_odd_window(self) -> None:
        """Test avec fenetre impaire."""
        # window=5, mid=2
        # Premiere moitie (head 2): 0, 0 = 0.0
        # Seconde moitie (tail 2): 1, 1 = 1.0
        df = pd.DataFrame({"resultat": [0.0, 0.0, 0.5, 1.0, 1.0]})
        result = _calculate_tendance(df, "resultat", 5)
        assert result == "hausse"


# ==============================================================================
# TESTS: Edge Cases (ISO 29119-4: Error Guessing)
# ==============================================================================


class TestRecentFormEdgeCases:
    """Tests edge cases pour robustesse."""

    def test_player_both_colors(self) -> None:
        """Test joueur jouant blanc ET noir est agrege."""
        df = pd.DataFrame(
            [
                # 5 matchs blanc
                {
                    "blanc_nom": "Joueur X",
                    "noir_nom": "A1",
                    "resultat_blanc": 1.0,
                    "resultat_noir": 0.0,
                    "type_resultat": "victoire_blanc",
                    "date": f"2025-01-0{i}",
                }
                for i in range(1, 6)
            ]
            + [
                # 5 matchs noir
                {
                    "blanc_nom": f"A{i}",
                    "noir_nom": "Joueur X",
                    "resultat_blanc": 0.0,
                    "resultat_noir": 1.0,
                    "type_resultat": "victoire_noir",
                    "date": f"2025-01-1{i}",
                }
                for i in range(1, 6)
            ]
        )

        result = calculate_recent_form(df, window=5)

        # Joueur X apparait une seule fois (agrege)
        joueur_x = result[result["joueur_nom"] == "Joueur X"]
        assert len(joueur_x) == 1
        # Forme parfaite (1.0 blanc + 1.0 noir) / 2 = 1.0
        assert joueur_x.iloc[0]["forme_recente"] == 1.0

    def test_all_same_results(self) -> None:
        """Test avec tous les resultats identiques (toutes victoires)."""
        df = pd.DataFrame(
            [
                {
                    "blanc_nom": "Joueur Y",
                    "noir_nom": f"Adversaire {i}",
                    "resultat_blanc": 1.0,
                    "resultat_noir": 0.0,
                    "type_resultat": "victoire_blanc",
                    "date": f"2025-01-{i:02d}",
                }
                for i in range(1, 8)
            ]
        )

        result = calculate_recent_form(df, window=5)

        joueur_y = result[result["joueur_nom"] == "Joueur Y"]
        assert joueur_y.iloc[0]["forme_recente"] == 1.0
        assert joueur_y.iloc[0]["forme_tendance"] == "stable"

    def test_all_draws(self) -> None:
        """Test avec tous les resultats nuls."""
        df = pd.DataFrame(
            [
                {
                    "blanc_nom": "Joueur Z",
                    "noir_nom": f"Adversaire {i}",
                    "resultat_blanc": 0.5,
                    "resultat_noir": 0.5,
                    "type_resultat": "nulle",
                    "date": f"2025-01-{i:02d}",
                }
                for i in range(1, 8)
            ]
        )

        result = calculate_recent_form(df, window=5)

        joueur_z = result[result["joueur_nom"] == "Joueur Z"]
        assert joueur_z.iloc[0]["forme_recente"] == 0.5
        assert joueur_z.iloc[0]["forme_tendance"] == "stable"

    def test_without_date_column(self) -> None:
        """Test sans colonne date (pas de tri)."""
        df = pd.DataFrame(
            [
                {
                    "blanc_nom": "Joueur W",
                    "noir_nom": f"Adversaire {i}",
                    "resultat_blanc": 1.0 if i % 2 == 0 else 0.0,
                    "resultat_noir": 0.0 if i % 2 == 0 else 1.0,
                    "type_resultat": "victoire_blanc" if i % 2 == 0 else "victoire_noir",
                }
                for i in range(1, 8)
            ]
        )

        result = calculate_recent_form(df, window=5)

        # Doit fonctionner meme sans colonne date
        assert not result.empty
        assert "Joueur W" in result["joueur_nom"].values
