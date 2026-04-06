"""Tests club_level — vases communiquants features — ISO 29119.

Document ID: ALICE-TEST-CLUB-LEVEL
Version: 1.0.0
Tests count: 8

Validates:
- team hierarchy ranking within club
- club_nb_teams count
- reinforcement_rate detection
- joueur_relegue / joueur_promu flags
- player_team_elo_gap sign

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML (forfait exclusion)
- ISO/IEC 5055:2021 - Code Quality (<300 lines)
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.features.club_level import (
    extract_club_level_features,
    extract_player_team_context,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(
    equipe_dom: str,
    equipe_ext: str,
    blanc_nom: str,
    noir_nom: str,
    saison: int = 2025,
    ronde: int = 1,
    blanc_elo: float = 1500.0,
    noir_elo: float = 1400.0,
    resultat_blanc: float = 1.0,
    type_resultat: str = "victoire_blanc",
) -> dict:
    return {
        "saison": saison,
        "ronde": ronde,
        "equipe_dom": equipe_dom,
        "equipe_ext": equipe_ext,
        "blanc_nom": blanc_nom,
        "noir_nom": noir_nom,
        "blanc_elo": blanc_elo,
        "noir_elo": noir_elo,
        "resultat_blanc": resultat_blanc,
        "type_resultat": type_resultat,
    }


@pytest.fixture()
def three_team_club() -> pd.DataFrame:
    """Marseille Echecs has 3 teams: N1, N2, Regionale.

    FFE-realistic naming: 'Marseille Echecs 1 N1', etc.
    Opponents use a different club name to avoid polluting the club grouping.
    """
    rows = []
    # Marseille N1 vs some opponent (round 1, 2)
    rows.append(_make_row("Marseille Echecs 1 N1", "Rival Club N1", "Alpha", "Zeta", ronde=1))
    rows.append(_make_row("Marseille Echecs 1 N1", "Rival Club N1", "Beta", "Eta", ronde=2))
    # Marseille N2 (round 1, 2)
    rows.append(
        _make_row(
            "Marseille Echecs 2 N2", "Other Club N2", "Gamma", "Theta", ronde=1, blanc_elo=1300.0
        )
    )
    rows.append(
        _make_row(
            "Marseille Echecs 2 N2", "Other Club N2", "Delta", "Iota", ronde=2, blanc_elo=1250.0
        )
    )
    # Marseille Regionale (round 1, 2)
    rows.append(
        _make_row(
            "Marseille Echecs 3 Regionale",
            "Divers Club Regionale",
            "Epsilon",
            "Kappa",
            ronde=1,
            blanc_elo=1100.0,
        )
    )
    rows.append(
        _make_row(
            "Marseille Echecs 3 Regionale",
            "Divers Club Regionale",
            "Zeta",
            "Lambda",
            ronde=2,
            blanc_elo=1050.0,
        )
    )
    return pd.DataFrame(rows)


@pytest.fixture()
def reinforcement_club() -> pd.DataFrame:
    """Club with player 'Renfort' playing for N1 in round 1, then N2 in round 2."""
    rows = [
        # Round 1: Renfort plays for N1
        _make_row("Lyon Echecs 1 N1", "Rival A N1", "Renfort", "Opp1", ronde=1, blanc_elo=1800.0),
        # Round 2: Renfort drops to N2 (reinforcement)
        _make_row("Lyon Echecs 2 N2", "Rival B N2", "Renfort", "Opp2", ronde=2, blanc_elo=1800.0),
        # Other N2 regular player (both rounds)
        _make_row("Lyon Echecs 2 N2", "Rival B N2", "Regular", "Opp3", ronde=1, blanc_elo=1200.0),
        _make_row("Lyon Echecs 2 N2", "Rival B N2", "Regular", "Opp4", ronde=2, blanc_elo=1200.0),
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def promo_club() -> pd.DataFrame:
    """Player 'Talent' mainly plays N3 but appears in N1 once (promotion)."""
    rows = [
        # Talent's primary team: N3 (3 rounds)
        _make_row("Paris Echecs 3 N3", "Adv Club N3", "Talent", "Opp1", ronde=1, blanc_elo=1600.0),
        _make_row("Paris Echecs 3 N3", "Adv Club N3", "Talent", "Opp2", ronde=2, blanc_elo=1600.0),
        _make_row("Paris Echecs 3 N3", "Adv Club N3", "Talent", "Opp3", ronde=3, blanc_elo=1600.0),
        # Talent promoted to N1 once
        _make_row("Paris Echecs 1 N1", "Adv Club N1", "Talent", "Opp4", ronde=4, blanc_elo=1600.0),
        # N1 regular player
        _make_row("Paris Echecs 1 N1", "Adv Club N1", "Expert", "Opp5", ronde=1, blanc_elo=2000.0),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: extract_club_level_features
# ---------------------------------------------------------------------------


class TestClubNbTeams:
    """Test club_nb_teams reflects correct team count."""

    def test_club_nb_teams(self, three_team_club: pd.DataFrame) -> None:
        """Club with 3 teams returns club_nb_teams=3 for all teams."""
        result = extract_club_level_features(three_team_club)
        assert not result.empty, "Result must not be empty"
        assert "club_nb_teams" in result.columns

        marseille_rows = result[result["equipe"].str.startswith("Marseille Echecs")]
        assert len(marseille_rows) == 3, f"Expected 3 Marseille rows, got:\n{result}"
        assert (marseille_rows["club_nb_teams"] == 3).all()

    def test_returns_expected_columns(self, three_team_club: pd.DataFrame) -> None:
        """Output contains all documented columns."""
        result = extract_club_level_features(three_team_club)
        expected_cols = {
            "equipe",
            "saison",
            "team_rank_in_club",
            "club_nb_teams",
            "reinforcement_rate",
            "stabilite_effectif",
            "elo_moyen_evolution",
        }
        assert expected_cols.issubset(set(result.columns))


class TestTeamRankOrdering:
    """Test team_rank_in_club ordering."""

    def test_team_rank_ordering(self, three_team_club: pd.DataFrame) -> None:
        """N1 ranked 1, N2 ranked 2, Regionale ranked 3."""
        result = extract_club_level_features(three_team_club)
        assert not result.empty

        rank_n1 = result[result["equipe"] == "Marseille Echecs 1 N1"]["team_rank_in_club"].iloc[0]
        rank_n2 = result[result["equipe"] == "Marseille Echecs 2 N2"]["team_rank_in_club"].iloc[0]
        rank_reg = result[result["equipe"] == "Marseille Echecs 3 Regionale"][
            "team_rank_in_club"
        ].iloc[0]

        assert (
            rank_n1 < rank_n2 < rank_reg
        ), f"Expected N1({rank_n1}) < N2({rank_n2}) < Reg({rank_reg})"

    def test_empty_df_returns_empty(self) -> None:
        """Empty input returns empty DataFrame gracefully."""
        result = extract_club_level_features(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# Tests: extract_player_team_context
# ---------------------------------------------------------------------------


class TestJoueurRelegue:
    """Test joueur_relegue: player descends to lower-level team."""

    def test_joueur_relegue_detected(self, reinforcement_club: pd.DataFrame) -> None:
        """Renfort's primary team is N1; appearance in N2 → joueur_relegue=True."""
        result = extract_player_team_context(reinforcement_club)
        assert not result.empty

        renfort_n2 = result[
            (result["joueur_nom"] == "Renfort") & (result["equipe"] == "Lyon Echecs 2 N2")
        ]
        assert len(renfort_n2) > 0, "Renfort must have rows for Lyon N2"
        assert renfort_n2[
            "joueur_relegue"
        ].all(), "Renfort appearing in N2 when primary=N1 must be flagged as relegue"

    def test_regular_player_not_relegue(self, reinforcement_club: pd.DataFrame) -> None:
        """Regular player with single team must NOT be flagged."""
        result = extract_player_team_context(reinforcement_club)
        regular_rows = result[result["joueur_nom"] == "Regular"]
        assert not regular_rows["joueur_relegue"].any()
        assert not regular_rows["joueur_promu"].any()


class TestJoueurPromu:
    """Test joueur_promu: player ascends to higher-level team."""

    def test_joueur_promu_detected(self, promo_club: pd.DataFrame) -> None:
        """Talent's primary team is N3; appearance in N1 → joueur_promu=True."""
        result = extract_player_team_context(promo_club)
        assert not result.empty

        talent_n1 = result[
            (result["joueur_nom"] == "Talent") & (result["equipe"] == "Paris Echecs 1 N1")
        ]
        assert len(talent_n1) > 0, "Talent must have rows for Paris N1"
        assert talent_n1[
            "joueur_promu"
        ].all(), "Talent appearing in N1 when primary=N3 must be flagged as promu"


class TestPlayerTeamEloGap:
    """Test player_team_elo_gap sign convention."""

    def test_player_team_elo_gap_positive(self, reinforcement_club: pd.DataFrame) -> None:
        """Renfort (Elo 1800) in N2 (avg ~1200-1800) → gap is positive (stronger than avg)."""
        result = extract_player_team_context(reinforcement_club)
        renfort_n2 = result[
            (result["joueur_nom"] == "Renfort") & (result["equipe"] == "Lyon Echecs 2 N2")
        ]
        # Renfort Elo=1800; N2 avg (Renfort 1800 + Regular 1200) / 2 = 1500
        # gap = 1800 - 1500 = 300 > 0
        assert (
            renfort_n2["player_team_elo_gap"] > 0
        ).all(), "Renfort (strong) playing for weaker team should have positive elo gap"
