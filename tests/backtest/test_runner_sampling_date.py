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


def _df(date_col: object = None) -> pd.DataFrame:
    """Build a minimal echiquiers DataFrame for sampling tests.

    @param date_col: value for the ``date`` column. Defaults to a
        datetime64 column of 2024-11-17, matching production schema.
    """
    if date_col is None:
        date_col = pd.to_datetime(["2024-11-17"] * 8)
    return pd.DataFrame(
        {
            "saison": [2024] * 8,
            "type_competition": ["national"] * 8,
            "division": ["Nationale 3"] * 8,
            "ronde": [3] * 8,
            "equipe_dom": ["A 3"] * 8,
            "equipe_ext": ["B 3"] * 8,
            "groupe": [""] * 8,
            "date": date_col,
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


def test_candidate_date_nat_yields_empty_string() -> None:
    """Regression: pd.NaT in date column must not leak "NaT" into MatchCandidate.

    echiquiers.parquet has 83 218 NaT dates. Before the fix, str(pd.NaT) == "NaT"
    slipped past the "nan" guard and stored "NaT" as the date, causing T9.2
    to silently degrade to Phase 3. After the fix (pd.notna), date must be "".
    """
    nat_col = pd.to_datetime([None] * 8)  # dtype datetime64[ns], all NaT
    cfg = RunnerConfig(
        saison=2024,
        rondes=(3,),
        max_matches=10,
        team_size=8,
        division="N3",
        division_filter="Nationale 3",
        type_competition="national",
    )
    cands = enumerate_candidates(_Cache(_df(date_col=nat_col)), cfg)
    assert len(cands) == 1
    assert cands[0].date == "", (
        f"Expected empty string for NaT date, got {cands[0].date!r}. "
        "The pd.NaT guard (pd.notna) may be missing."
    )
