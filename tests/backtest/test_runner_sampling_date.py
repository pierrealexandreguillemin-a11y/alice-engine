"""T9.1 — MatchCandidate carries real match date from echiquiers (Phase 4a).

Document ID: ALICE-BACKTEST-T9.1
Version: 1.0.0
"""

from __future__ import annotations

import pandas as pd

from scripts.backtest.runner_sampling import enumerate_candidates
from scripts.backtest.runner_types import RunnerConfig


class _Cache:
    def __init__(self, df: pd.DataFrame) -> None:
        self.echiquiers_total = df
        self.team_to_club = {"A 3": "A", "B 3": "B"}
        self.joueurs_by_club = {
            "A": pd.DataFrame({"nr_ffe": range(8)}),
            "B": pd.DataFrame({"nr_ffe": range(8)}),
        }


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "saison": [2024] * 8,
            "type_competition": ["national"] * 8,
            "division": ["Nationale 3"] * 8,
            "ronde": [3] * 8,
            "equipe_dom": ["A 3"] * 8,
            "equipe_ext": ["B 3"] * 8,
            "groupe": [""] * 8,
            "date": ["2024-11-17"] * 8,
        }
    )


def test_candidate_carries_match_date() -> None:
    cfg = RunnerConfig(
        saison=2024,
        rondes=(3,),
        max_matches=10,
        team_size=8,
        division="N3",
        division_filter="Nationale 3",
        type_competition="national",
    )
    cands = enumerate_candidates(_Cache(_df()), cfg)
    assert len(cands) == 1
    assert cands[0].date == "2024-11-17"
