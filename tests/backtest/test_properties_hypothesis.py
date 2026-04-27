"""T19 Property-based tests via Hypothesis (Plan 3 V2 §T19).

15 property tests sur metrics T13-T17 (DoD verbatim) : top_k_recall (3),
accuracy_at_k (2), jaccard_max (2), brier_presence (2), brier_skill_score
(1), ece_presence (1), bootstrap_ci (2), mcnemar_paired (1), ScenarioSet
weights sum invariant (1). Symmetries, bounds, invariants.

Sources : ISO 29119 §6.4 property-based testing, MacIver 2019 Hypothesis,
Hughes & Claessen 2000 QuickCheck (ICFP).

Document ID: ALICE-BACKTEST-PROPERTIES
Version: 1.0.0
"""

from __future__ import annotations

import string

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

from scripts.backtest.bootstrap import bootstrap_ci
from scripts.backtest.calibration import ece_presence
from scripts.backtest.ground_truth import ObservedLineup, ObservedPlayer
from scripts.backtest.metrics import (
    accuracy_at_k,
    brier_presence,
    brier_skill_score,
    jaccard_max,
    top_k_recall,
)
from scripts.backtest.statistical import mcnemar_paired
from services.ali.scenario import BoardAssignment, Lineup, Scenario, ScenarioSet
from services.ali.types import PlayerCandidate

_NOM = st.text(alphabet=string.ascii_uppercase, min_size=3, max_size=8)
_PRENOM = st.text(alphabet=string.ascii_letters, min_size=3, max_size=8)


@st.composite
def _name_pair(draw: st.DrawFn) -> tuple[str, str]:
    """Draw (NOM, Prenom). Canonical name = f'{NOM} {Prenom}' (metrics convention)."""
    return draw(_NOM), draw(_PRENOM)


def _pairs_from_obs(obs: ObservedLineup) -> list[tuple[str, str]]:
    """Reverse-engineer (nom, prenom) pairs from canonical 'NOM Prenom' joueur_nom."""
    return [
        (p.joueur_nom.split(" ", 1)[0], p.joueur_nom.split(" ", 1)[1])
        for p in obs.players
        if " " in p.joueur_nom
    ]


@st.composite
def _observed_lineup(draw: st.DrawFn) -> ObservedLineup:
    """ObservedLineup of 1-8 distinct (name, board) tuples."""
    pairs = draw(st.lists(_name_pair(), min_size=1, max_size=8, unique=True))
    players = tuple(
        ObservedPlayer(joueur_nom=f"{n} {p}".strip(), echiquier=i + 1, elo=1500)
        for i, (n, p) in enumerate(pairs)
    )
    return ObservedLineup(club_name="C", saison=2024, ronde=5, players=players)


_PLAYER_DEFAULTS: dict[str, object] = {
    "elo": 1500,
    "club": "C",
    "mute": False,
    "genre": "M",
    "categorie": "SE",
    "licence_active": True,
}


def _make_player(nom: str, prenom: str, ffe: str = "0") -> PlayerCandidate:
    return PlayerCandidate(nr_ffe=ffe, nom=nom, prenom=prenom, **_PLAYER_DEFAULTS)  # type: ignore[arg-type]


def _make_scenario(
    pairs: list[tuple[str, str]], weight: float, joint_prob: float = 0.1
) -> Scenario:
    """Scenario from name pairs (board = index+1)."""
    assignments = tuple(
        BoardAssignment(board=i + 1, player=_make_player(n, p, ffe=str(i)), p_assignment=0.5)
        for i, (n, p) in enumerate(pairs)
    )
    return Scenario(
        lineup=Lineup(team_size=len(assignments), assignments=assignments),
        joint_prob=joint_prob,
        weight=weight,
        source="topk",
    )


def _make_set(scenarios: list[Scenario]) -> ScenarioSet:
    """ScenarioSet without validate() (allows arbitrary length)."""
    return ScenarioSet(
        scenarios=tuple(scenarios),
        opponent_club_id="C",
        round_date="2024-09-01",
        generated_at="2024-09-01T00:00:00Z",
        lineage_hash="0" * 64,
    )


@st.composite
def _scenario_set_normalized(draw: st.DrawFn) -> tuple[ScenarioSet, ObservedLineup]:
    """(scenario_set, observed) sharing a name pool. weights sum=1."""
    pool = draw(st.lists(_name_pair(), min_size=2, max_size=10, unique=True))
    n_scen = draw(st.integers(min_value=1, max_value=5))
    raw_w = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
            min_size=n_scen,
            max_size=n_scen,
        )
    )
    total = sum(raw_w)
    weights = [w / total for w in raw_w]
    scenarios = []
    for i, w in enumerate(weights):
        size = draw(st.integers(min_value=1, max_value=len(pool)))
        sub = draw(st.lists(st.sampled_from(pool), min_size=size, max_size=size, unique=True))
        scenarios.append(_make_scenario(sub, w, joint_prob=0.1 + 0.01 * i))
    obs_size = draw(st.integers(min_value=1, max_value=len(pool)))
    obs_pairs = draw(
        st.lists(st.sampled_from(pool), min_size=obs_size, max_size=obs_size, unique=True)
    )
    obs_players = tuple(
        ObservedPlayer(joueur_nom=f"{n} {p}".strip(), echiquier=i + 1, elo=1500)
        for i, (n, p) in enumerate(obs_pairs)
    )
    obs = ObservedLineup(club_name="C", saison=2024, ronde=5, players=obs_players)
    return _make_set(scenarios), obs


_HSETTINGS = settings(
    max_examples=50,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
_VALUES_STRAT = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=50,
)
_BOOL_LIST = st.lists(st.booleans(), min_size=1, max_size=50)


@_HSETTINGS
@given(_scenario_set_normalized())
def test_property_recall_in_unit_interval(args: tuple[ScenarioSet, ObservedLineup]) -> None:
    """Recall always in [0, 1]."""
    ss, obs = args
    assert 0.0 <= top_k_recall(obs, ss) <= 1.0


@_HSETTINGS
@given(_observed_lineup())
def test_property_recall_identity_one(obs: ObservedLineup) -> None:
    """recall(obs, scenarios containing exactly obs) == 1.0."""
    pairs = _pairs_from_obs(obs)
    if not pairs:
        return
    ss = _make_set([_make_scenario(pairs, weight=1.0)])
    assert top_k_recall(obs, ss) == 1.0


@_HSETTINGS
@given(_observed_lineup(), _observed_lineup())
def test_property_recall_disjoint_is_zero(obs: ObservedLineup, other: ObservedLineup) -> None:
    """recall(obs, disjoint scenarios) == 0.0."""
    if obs.player_names() & other.player_names():
        return
    pairs = _pairs_from_obs(other)
    if not pairs:
        return
    ss = _make_set([_make_scenario(pairs, weight=1.0)])
    assert top_k_recall(obs, ss) == 0.0


@_HSETTINGS
@given(_scenario_set_normalized())
def test_property_accuracy_in_unit_interval(args: tuple[ScenarioSet, ObservedLineup]) -> None:
    """accuracy_at_k in [0, 1]."""
    ss, obs = args
    assert 0.0 <= accuracy_at_k(obs, ss) <= 1.0


@_HSETTINGS
@given(_observed_lineup())
def test_property_accuracy_identity_one(obs: ObservedLineup) -> None:
    """accuracy_at_k(obs, top scenario = obs) == 1.0."""
    pairs = _pairs_from_obs(obs)
    if not pairs:
        return
    ss = _make_set([_make_scenario(pairs, weight=1.0)])
    assert accuracy_at_k(obs, ss) == 1.0


@_HSETTINGS
@given(_scenario_set_normalized())
def test_property_jaccard_in_unit_interval(args: tuple[ScenarioSet, ObservedLineup]) -> None:
    """jaccard_max in [0, 1]."""
    ss, obs = args
    assert 0.0 <= jaccard_max(obs, ss) <= 1.0


@_HSETTINGS
@given(_observed_lineup())
def test_property_jaccard_identity_one(obs: ObservedLineup) -> None:
    """jaccard_max(obs, scenarios = obs) == 1.0."""
    pairs = _pairs_from_obs(obs)
    if not pairs:
        return
    ss = _make_set([_make_scenario(pairs, weight=1.0)])
    assert jaccard_max(obs, ss) == 1.0


@_HSETTINGS
@given(_scenario_set_normalized())
def test_property_brier_in_unit_interval(args: tuple[ScenarioSet, ObservedLineup]) -> None:
    """Brier in [0, 1] (probas in [0,1] -> max squared diff = 1)."""
    ss, obs = args
    assert 0.0 <= brier_presence(obs, ss) <= 1.0


@_HSETTINGS
@given(_observed_lineup())
def test_property_brier_perfect_is_zero(obs: ObservedLineup) -> None:
    """Single scenario weight=1 == observed -> brier = 0."""
    pairs = _pairs_from_obs(obs)
    if not pairs:
        return
    ss = _make_set([_make_scenario(pairs, weight=1.0)])
    assert brier_presence(obs, ss) == pytest.approx(0.0, abs=1e-9)


@_HSETTINGS
@given(_scenario_set_normalized(), st.floats(min_value=0.01, max_value=1.0))
def test_property_bss_upper_bounded(
    args: tuple[ScenarioSet, ObservedLineup], baseline: float
) -> None:
    """BSS = 1 - brier/baseline <= 1.0."""
    ss, obs = args
    assert brier_skill_score(obs, ss, baseline) <= 1.0


@_HSETTINGS
@given(_scenario_set_normalized())
def test_property_ece_non_negative(args: tuple[ScenarioSet, ObservedLineup]) -> None:
    """ECE >= 0."""
    ss, obs = args
    assert ece_presence(obs, ss) >= 0.0


@_HSETTINGS
@given(_VALUES_STRAT)
def test_property_bootstrap_containment(values: list[float]) -> None:
    """ci.lower <= ci.point <= ci.upper."""
    ci = bootstrap_ci(values, n_resamples=100, seed=42)
    assert ci.lower <= ci.point <= ci.upper


@_HSETTINGS
@given(_VALUES_STRAT)
def test_property_bootstrap_point_is_mean(values: list[float]) -> None:
    """ci.point == mean(values)."""
    ci = bootstrap_ci(values, n_resamples=100, seed=42)
    assert ci.point == pytest.approx(sum(values) / len(values), abs=1e-9)


@_HSETTINGS
@given(
    _BOOL_LIST.flatmap(
        lambda x: st.tuples(st.just(x), st.lists(st.booleans(), min_size=len(x), max_size=len(x)))
    )
)
def test_property_mcnemar_pvalue_in_unit_interval(pair: tuple[list[bool], list[bool]]) -> None:
    """mcnemar.p_value in [0, 1]."""
    a, b = pair
    if len(a) != len(b):
        return
    res = mcnemar_paired(a, b)
    assert 0.0 <= res.p_value <= 1.0


@_HSETTINGS
@given(
    st.lists(st.floats(min_value=0.01, max_value=1.0, allow_nan=False), min_size=20, max_size=20)
)
def test_property_scenario_set_weights_sum_invariant(raw_weights: list[float]) -> None:
    """ScenarioSet.validate() accepts sum(weights)==1.0 (after normalize)."""
    total = sum(raw_weights)
    weights = [w / total for w in raw_weights]
    pairs = [("AAA", f"P{i:02d}") for i in range(20)]
    scenarios = [_make_scenario([pairs[i]], w) for i, w in enumerate(weights)]
    _make_set(scenarios).validate()
