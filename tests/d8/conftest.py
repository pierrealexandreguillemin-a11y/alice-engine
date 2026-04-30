"""Shared fixtures for D8 tests (ISO 29119)."""

from __future__ import annotations

import pytest

from scripts.backtest.runner_types import MatchStats


@pytest.fixture
def dummy_match() -> MatchStats:
    return MatchStats(
        saison=2024,
        ronde=5,
        user_team="USR",
        opponent_team="OPP",
        recall_ali=0.80,
        accuracy_ali=0.85,
        jaccard_ali=0.70,
        brier_ali=0.20,
        ece_ali=0.04,
        recall_baseline=0.40,
        brier_baseline=0.50,
        bss=0.60,
        e_score_predicted=4.0,
        e_score_observed=4.5,
        e_score_mae=0.5,
        ali_correct=True,
        baseline_correct=False,
    )


@pytest.fixture
def matches_5() -> list[MatchStats]:
    return [
        MatchStats(
            saison=2024,
            ronde=r,
            user_team=f"USR{r}",
            opponent_team=f"OPP{r}",
            recall_ali=0.50 + r * 0.1,
            accuracy_ali=0.85,
            jaccard_ali=0.60,
            brier_ali=0.20,
            ece_ali=0.04,
            recall_baseline=0.40,
            brier_baseline=0.50,
            bss=0.60,
            e_score_predicted=4.0,
            e_score_observed=4.5,
            e_score_mae=0.5,
            ali_correct=r > 2,
            baseline_correct=False,
        )
        for r in range(1, 6)
    ]
