"""Tests Edge Cases - ISO 29119.

Document ID: ALICE-TEST-FE-EDGE
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import numpy as np
import pandas as pd

from scripts.features.standings import calculate_standings


class TestEdgeCases:
    """Tests edge cases ISO 29119."""

    def test_nan_scores(self) -> None:
        """Test avec scores NaN."""
        df = pd.DataFrame(
            [
                {
                    "saison": 2025,
                    "competition": "Test",
                    "division": "N1",
                    "groupe": "A",
                    "ronde": 1,
                    "equipe_dom": "A",
                    "equipe_ext": "B",
                    "score_dom": np.nan,
                    "score_ext": np.nan,
                }
            ]
        )
        # Ne doit pas crasher
        result = calculate_standings(df)
        # Peut etre vide ou ignorer les NaN
        assert isinstance(result, pd.DataFrame)

    def test_single_ronde(self) -> None:
        """Test avec une seule ronde."""
        df = pd.DataFrame(
            [
                {
                    "saison": 2025,
                    "competition": "Test",
                    "division": "N1",
                    "groupe": "A",
                    "ronde": 1,
                    "equipe_dom": "A",
                    "equipe_ext": "B",
                    "score_dom": 4,
                    "score_ext": 2,
                }
            ]
        )
        result = calculate_standings(df)
        assert not result.empty
        assert len(result) == 2  # 2 equipes

    def test_multi_saison(self, sample_matches: pd.DataFrame) -> None:
        """Test avec plusieurs saisons."""
        df2 = sample_matches.copy()
        df2["saison"] = 2024

        df_combined = pd.concat([sample_matches, df2])
        result = calculate_standings(df_combined)

        assert not result.empty
        assert 2024 in result["saison"].values
        assert 2025 in result["saison"].values

    def test_tiebreaker_same_points(self) -> None:
        """Test tie-breaker quand meme nombre de points."""
        df = pd.DataFrame(
            [
                {
                    "saison": 2025,
                    "competition": "Test",
                    "division": "N1",
                    "groupe": "A",
                    "ronde": 1,
                    "equipe_dom": "Equipe A",
                    "equipe_ext": "Equipe B",
                    "score_dom": 4,
                    "score_ext": 2,
                },
                {
                    "saison": 2025,
                    "competition": "Test",
                    "division": "N1",
                    "groupe": "A",
                    "ronde": 1,
                    "equipe_dom": "Equipe C",
                    "equipe_ext": "Equipe D",
                    "score_dom": 4,
                    "score_ext": 2,
                },
                {
                    "saison": 2025,
                    "competition": "Test",
                    "division": "N1",
                    "groupe": "A",
                    "ronde": 2,
                    "equipe_dom": "Equipe A",
                    "equipe_ext": "Equipe C",
                    "score_dom": 5,
                    "score_ext": 1,
                },
                {
                    "saison": 2025,
                    "competition": "Test",
                    "division": "N1",
                    "groupe": "A",
                    "ronde": 2,
                    "equipe_dom": "Equipe B",
                    "equipe_ext": "Equipe D",
                    "score_dom": 4,
                    "score_ext": 2,
                },
            ]
        )

        result = calculate_standings(df)
        r2 = result[result["ronde"] == 2]

        # A: 4 pts, C: 2 pts, B: 2 pts, D: 0 pts
        a_pos = r2[r2["equipe"] == "Equipe A"]["position"].values[0]
        assert a_pos == 1

        b_pos = r2[r2["equipe"] == "Equipe B"]["position"].values[0]
        c_pos = r2[r2["equipe"] == "Equipe C"]["position"].values[0]
        # Les deux ont 2 pts, mais un doit etre 2e et l'autre 3e
        assert {b_pos, c_pos} == {2, 3}
