"""Tests Recent Form W/D/L decomposition + stratification - ISO 29119.

Document ID: ALICE-TEST-FE-RECENT-FORM
Version: 1.0.0
Tests count: 5

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)
- ISO/IEC 42001:2023 - AI traceability (W/D/L, stratification)
- ISO/IEC 5259:2024 - Data Quality (forfait exclusion)

Author: ALICE Engine Team
Last Updated: 2026-03-22
"""

import pandas as pd
import pytest

from scripts.features.recent_form import calculate_recent_form


@pytest.fixture
def games_wdl_decomposition() -> pd.DataFrame:
    """5 parties: 2W + 2D + 1L => win_rate=0.4, draw_rate=0.4, expected=0.6."""
    return pd.DataFrame(
        [
            {
                "blanc_nom": "Alpha",
                "noir_nom": f"Adv{i}",
                "resultat_blanc": r,
                "resultat_noir": 1.0 - r,
                "type_resultat": t,
                "date": f"2025-01-{i:02d}",
            }
            for i, (r, t) in enumerate(
                [
                    (1.0, "victoire_blanc"),  # W
                    (1.0, "victoire_blanc"),  # W
                    (0.5, "nulle"),  # D
                    (0.5, "nulle"),  # D
                    (0.0, "victoire_noir"),  # L
                ],
                start=1,
            )
        ]
    )


@pytest.fixture
def games_stratified() -> pd.DataFrame:
    """Player with 3 national games (0W 3D) + 2 regional games (2W 0D)."""
    rows = []
    # 3 national games — all draws
    for i in range(3):
        rows.append(
            {
                "blanc_nom": "Beta",
                "noir_nom": f"NatAdv{i}",
                "resultat_blanc": 0.5,
                "resultat_noir": 0.5,
                "type_resultat": "nulle",
                "type_competition": "national",
                "date": f"2025-01-{i + 1:02d}",
            }
        )
    # 2 regional games — all wins (not enough for stratification threshold=3)
    for i in range(2):
        rows.append(
            {
                "blanc_nom": "Beta",
                "noir_nom": f"RegAdv{i}",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "type_competition": "regional",
                "date": f"2025-02-{i + 1:02d}",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def games_with_forfeits() -> pd.DataFrame:
    """6 games: 5 played wins + 1 forfait that must be excluded."""
    rows = []
    for i in range(1, 6):
        rows.append(
            {
                "blanc_nom": "Gamma",
                "noir_nom": f"Adv{i}",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": f"2025-01-{i:02d}",
            }
        )
    # Forfait — must be excluded
    rows.append(
        {
            "blanc_nom": "Gamma",
            "noir_nom": "ForfeitAdv",
            "resultat_blanc": 2.0,
            "resultat_noir": 0.0,
            "type_resultat": "forfait_noir",
            "date": "2025-02-01",
        }
    )
    return pd.DataFrame(rows)


class TestWDLFormDecomposition:
    """Tests decomposition W/D/L de la forme recente."""

    def test_wdl_form_decomposition(self, games_wdl_decomposition: pd.DataFrame) -> None:
        """Recent form must return win_rate, draw_rate, expected_score.

        5 games: 2 wins, 2 draws, 1 loss
        => win_rate=0.4, draw_rate=0.4, expected_score=0.6
        """
        result = calculate_recent_form(games_wdl_decomposition, window=5)

        alpha = result[result["joueur_nom"] == "Alpha"]
        assert len(alpha) == 1, "Alpha doit avoir exactement une ligne"

        row = alpha.iloc[0]
        assert (
            abs(row["win_rate_recent"] - 0.4) < 1e-9
        ), f"win_rate attendu 0.4, obtenu {row['win_rate_recent']}"
        assert (
            abs(row["draw_rate_recent"] - 0.4) < 1e-9
        ), f"draw_rate attendu 0.4, obtenu {row['draw_rate_recent']}"
        assert (
            abs(row["expected_score_recent"] - 0.6) < 1e-9
        ), f"expected_score attendu 0.6, obtenu {row['expected_score_recent']}"

    def test_wdl_no_conflation_draws_losses(self) -> None:
        """draw_rate et win_rate doivent etre independants.

        Joueur 1: 2W+1D+2L => win=0.4, draw=0.2, expected=0.5
        Joueur 2: 1W+3D+1L => win=0.2, draw=0.6, expected=0.5
        Meme expected_score mais profils differents.
        """
        rows = []
        # Joueur 1: 2W + 1D + 2L
        for r, t in [
            (1.0, "victoire_blanc"),
            (1.0, "victoire_blanc"),
            (0.5, "nulle"),
            (0.0, "victoire_noir"),
            (0.0, "victoire_noir"),
        ]:
            rows.append(
                {
                    "blanc_nom": "J1",
                    "noir_nom": "X",
                    "resultat_blanc": r,
                    "resultat_noir": 1.0 - r,
                    "type_resultat": t,
                }
            )
        # Joueur 2: 1W + 3D + 1L
        for r, t in [
            (1.0, "victoire_blanc"),
            (0.5, "nulle"),
            (0.5, "nulle"),
            (0.5, "nulle"),
            (0.0, "victoire_noir"),
        ]:
            rows.append(
                {
                    "blanc_nom": "J2",
                    "noir_nom": "Y",
                    "resultat_blanc": r,
                    "resultat_noir": 1.0 - r,
                    "type_resultat": t,
                }
            )

        result = calculate_recent_form(pd.DataFrame(rows), window=5)

        j1 = result[result["joueur_nom"] == "J1"].iloc[0]
        j2 = result[result["joueur_nom"] == "J2"].iloc[0]

        # Same expected_score but different win/draw profiles
        assert abs(j1["expected_score_recent"] - 0.5) < 1e-9
        assert abs(j2["expected_score_recent"] - 0.5) < 1e-9
        assert j1["win_rate_recent"] != j2["win_rate_recent"]
        assert j1["draw_rate_recent"] != j2["draw_rate_recent"]


class TestCompetitionStratification:
    """Tests stratification par type_competition."""

    def test_competition_stratification(self, games_stratified: pd.DataFrame) -> None:
        """Form must be computed within same competition level.

        Beta: 3 national (all draws) + 2 regional (all wins, < threshold).
        National form (>=3 games): win_rate=0, draw_rate=1.0
        Regional form (<3 games threshold): fallback to all games.
        """
        result = calculate_recent_form(games_stratified, window=5)

        beta_national = result[
            (result["joueur_nom"] == "Beta") & (result["type_competition"] == "national")
        ]
        assert len(beta_national) == 1, "Beta national doit avoir une ligne"

        row = beta_national.iloc[0]
        assert (
            row["win_rate_recent"] == 0.0
        ), f"win_rate national attendu 0.0, obtenu {row['win_rate_recent']}"
        assert (
            row["draw_rate_recent"] == 1.0
        ), f"draw_rate national attendu 1.0, obtenu {row['draw_rate_recent']}"

    def test_fallback_when_few_games_at_level(self, games_stratified: pd.DataFrame) -> None:
        """Regional (<3 games) falls back to all games for Beta."""
        result = calculate_recent_form(games_stratified, window=5)

        beta_regional = result[
            (result["joueur_nom"] == "Beta") & (result["type_competition"] == "regional")
        ]
        # Regional has 2 games → fallback; total 5 games (3 draws + 2 wins)
        assert len(beta_regional) == 1
        row = beta_regional.iloc[0]
        # Fallback uses all 5 games: 2W + 3D
        assert (
            abs(row["win_rate_recent"] - 0.4) < 1e-9
        ), f"win_rate fallback attendu 0.4, obtenu {row['win_rate_recent']}"


class TestForfeitsExcluded:
    """Tests exclusion forfaits de la forme recente."""

    def test_forfeits_excluded_from_form(self, games_with_forfeits: pd.DataFrame) -> None:
        """Forfait results (type_resultat=forfait_*) must not appear in form.

        Gamma: 5 real wins + 1 forfait.
        Expected: win_rate=1.0 (forfait exclu).
        """
        result = calculate_recent_form(games_with_forfeits, window=5)

        gamma = result[result["joueur_nom"] == "Gamma"]
        assert len(gamma) == 1
        assert gamma.iloc[0]["win_rate_recent"] == 1.0, "Le forfait ne doit pas contaminer win_rate"
        assert gamma.iloc[0]["draw_rate_recent"] == 0.0
