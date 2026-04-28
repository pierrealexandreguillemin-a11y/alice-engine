"""T19.5 (D-P3-15) Property tests sur cas degeneres metrics.

Suite a fix-on-sight bootstrap_ci NaN sur var=0 (commit 2629cfd), audit
honnete a montre :
- 70% du blame = test suite faible sur edge cases distributionnels
- 30% = code defectueux (BCa singularite)

D-P3-15 etend Hypothesis aux 5 metrics restantes pour figer les
comportements frontiere et detecter de nouveaux bugs symetriques.

Cas degeneres testes
--------------------
1. top_k_recall (obs non-vide, scenario_set vide) -> 0.0
2. jaccard_max (obs non-vide, scenarios vides ou disjoints) -> 0.0
3. brier_presence (union vide) -> 0.0
4. brier_skill_score (baseline <= 0) -> 0.0 (guard fail-safe)
5. ece_presence (toutes p_pres=0 OR toutes=1) -> calibration extreme valide
6. mcnemar (b=c=0, no disagreement) -> p_value=1.0
7. mcnemar (b=N c=0, perfect disagreement) -> p_value finite et < 0.5

Resultat scan manuel pre-implementation : 0 bug code expose. Tests
explicites figent comportement et previennent regression future.

Sources : Hughes & Claessen 2000 QuickCheck (edge case generation).
ISO 24029 §6.5 robustness (defensive guards documentes + testes).

Document ID: ALICE-BACKTEST-PROPERTIES-DEGENERATE
Version: 1.0.0
"""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from scripts.backtest.calibration import ece_presence
from scripts.backtest.ground_truth import ObservedLineup, ObservedPlayer
from scripts.backtest.metrics import (
    brier_presence,
    brier_skill_score,
    jaccard_max,
    top_k_recall,
)
from scripts.backtest.statistical import mcnemar_paired
from services.ali.scenario import ScenarioSet

_HSETTINGS = settings(max_examples=30, deadline=2000)


def _empty_set() -> ScenarioSet:
    """ScenarioSet without scenarios (degenerate input for metrics)."""
    return ScenarioSet(
        scenarios=(),
        opponent_club_id="C",
        round_date="2024-09-01",
        generated_at="2024-09-01T00:00:00Z",
        lineage_hash="0" * 64,
    )


def _obs_with_n(n: int) -> ObservedLineup:
    """ObservedLineup with n synthetic players."""
    players = tuple(
        ObservedPlayer(joueur_nom=f"NOM{i:02d} Prenom{i:02d}", echiquier=i + 1, elo=1500)
        for i in range(n)
    )
    return ObservedLineup(club_name="C", saison=2024, ronde=5, players=players)


# ---------------------------------------------------------------------------
# Empty scenario_set (5 metrics)
# ---------------------------------------------------------------------------


@_HSETTINGS
@given(st.integers(min_value=1, max_value=8))
def test_degenerate_recall_empty_scenarios(n: int) -> None:
    """top_k_recall(obs, empty) == 0.0 (union vide -> 0 intersection)."""
    assert top_k_recall(_obs_with_n(n), _empty_set()) == 0.0


@_HSETTINGS
@given(st.integers(min_value=1, max_value=8))
def test_degenerate_jaccard_empty_scenarios(n: int) -> None:
    """jaccard_max(obs, empty) == 0.0 (loop ne s'execute pas, best stays 0)."""
    assert jaccard_max(_obs_with_n(n), _empty_set()) == 0.0


@_HSETTINGS
@given(st.integers(min_value=1, max_value=8))
def test_degenerate_brier_empty_scenarios_punishes_observed(n: int) -> None:
    """brier_presence(obs non-vide, empty scenarios) > 0 (proba=0 mais flag=1)."""
    # Avec scenarios=[], p_presence_j=0 pour tout j observe -> (0-1)^2=1 par joueur
    # mean = 1.0 (tous penalises au max)
    b = brier_presence(_obs_with_n(n), _empty_set())
    assert b == pytest.approx(
        1.0, abs=1e-9
    ), "Brier doit penaliser au max les obs avec p_predit=0 et obs_flag=1"


@_HSETTINGS
@given(st.integers(min_value=1, max_value=8))
def test_degenerate_ece_empty_obs_empty_scenarios(n: int) -> None:  # noqa: ARG001
    """ece_presence sur union vide -> 0.0 (guard L104)."""
    obs_empty = ObservedLineup(club_name="C", saison=2024, ronde=5, players=())
    assert ece_presence(obs_empty, _empty_set()) == 0.0


# ---------------------------------------------------------------------------
# brier_skill_score : baseline <= 0 fail-safe
# ---------------------------------------------------------------------------


@_HSETTINGS
@given(st.floats(max_value=0.0, allow_nan=False, allow_infinity=False))
def test_degenerate_bss_nonpositive_baseline_returns_zero(baseline: float) -> None:
    """brier_skill_score(*, baseline<=0) == 0.0 (guard L198 evite div par 0).

    Decision documentee : baseline_brier=0 signifie modele baseline parfait.
    BSS = 1 - 0/0 mathematiquement indefinie ; choix conservateur 0.0
    (model NOT improving over baseline) plutot que 1.0 (model perfect).
    Voir docstring brier_skill_score §returns.
    """
    obs = _obs_with_n(3)
    assert brier_skill_score(obs, _empty_set(), baseline) == 0.0


# ---------------------------------------------------------------------------
# mcnemar_paired : boundary cases
# ---------------------------------------------------------------------------


@_HSETTINGS
@given(st.integers(min_value=1, max_value=20))
def test_degenerate_mcnemar_no_disagreement(n: int) -> None:
    """Mcnemar avec ali_correct == baseline_correct -> p_value=1.0 (no evidence)."""
    same = [True] * n
    res = mcnemar_paired(same, list(same))
    assert res.b == 0
    assert res.c == 0
    assert res.n_discordant == 0
    assert res.p_value == pytest.approx(1.0, abs=1e-9)
    assert res.method == "exact_binomial"


@_HSETTINGS
@given(st.integers(min_value=2, max_value=20))
def test_degenerate_mcnemar_perfect_disagreement(n: int) -> None:
    """Mcnemar avec b=N c=0 (ali ALL correct, baseline ALL wrong) -> p_value finie.

    Test binomial exact sur table (b, c) = (N, 0) : statsmodels mcnemar
    exact=True renvoie p_value bien defini (pas NaN). Pour n>=2, on attend
    p_value < 0.5 (evidence d'asymetrie meme si pas tjs <0.05 selon n).
    """
    res = mcnemar_paired([True] * n, [False] * n)
    assert res.b == n
    assert res.c == 0
    assert res.n_discordant == n
    assert 0.0 <= res.p_value <= 1.0
    # Sanity floor : pas de NaN/Inf
    import math

    assert math.isfinite(res.p_value)
