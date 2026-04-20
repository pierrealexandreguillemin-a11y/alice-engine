"""Tests T3 + T3b metrics (Top-K recall + Accuracy@K).

Plan 3 V2 Phase 3. ISO 29119 : coverage >= 80%, fixtures dédiées.
"""

from __future__ import annotations

import pytest

from scripts.backtest.ground_truth import ObservedLineup, ObservedPlayer
from scripts.backtest.metrics import accuracy_at_k, top_k_recall
from services.ali.scenario import BoardAssignment, Lineup, Scenario, ScenarioSet
from services.ali.types import PlayerCandidate


def _make_player(nom: str, prenom: str = "X", elo: int = 1800) -> PlayerCandidate:
    return PlayerCandidate(
        nr_ffe=f"{nom[0]}00001",
        nom=nom,
        prenom=prenom,
        elo=elo,
        club="C1",
        mute=False,
        genre="M",
        categorie="SE",
        licence_active=True,
    )


def _make_scenario(
    names: list[str],
    weight: float,
    source: str = "topk",
) -> Scenario:
    assignments = tuple(
        BoardAssignment(board=i + 1, player=_make_player(name), p_assignment=1.0)
        for i, name in enumerate(names)
    )
    lineup = Lineup(team_size=len(names), assignments=assignments)
    return Scenario(
        lineup=lineup,
        joint_prob=weight,
        weight=weight,
        source=source,  # type: ignore[arg-type]
    )


def _make_scenario_set(scenarios: list[Scenario]) -> ScenarioSet:
    return ScenarioSet(
        scenarios=tuple(scenarios),
        opponent_club_id="C1",
        round_date="2024-01-01",
        generated_at="2024-01-01T00:00:00Z",
        lineage_hash="a" * 64,
    )


def _make_observed(names: list[str]) -> ObservedLineup:
    players = tuple(
        ObservedPlayer(joueur_nom=f"{n} X", echiquier=i + 1, elo=1800) for i, n in enumerate(names)
    )
    return ObservedLineup(club_name="C1", saison=2024, ronde=5, players=players)


# --- T3 Top-K recall ---


def test_top_k_recall_all_observed_in_union() -> None:
    """Tous joueurs observed présents dans scenarios -> recall = 1.0."""
    observed = _make_observed(["ALPHA", "BETA", "GAMMA"])
    s1 = _make_scenario(["ALPHA", "BETA"], 0.5)
    s2 = _make_scenario(["GAMMA", "DELTA"], 0.5)
    ss = _make_scenario_set([s1, s2])
    assert top_k_recall(observed, ss) == 1.0


def test_top_k_recall_none_in_union() -> None:
    """Aucun observed dans scenarios -> recall = 0.0."""
    observed = _make_observed(["XYZ", "WWW"])
    s1 = _make_scenario(["ALPHA", "BETA"], 1.0)
    ss = _make_scenario_set([s1])
    assert top_k_recall(observed, ss) == 0.0


def test_top_k_recall_half_in_union() -> None:
    """Moitié des observed dans scenarios -> recall = 0.5."""
    observed = _make_observed(["ALPHA", "BETA", "XYZ", "WWW"])
    s1 = _make_scenario(["ALPHA", "BETA"], 1.0)
    ss = _make_scenario_set([s1])
    assert top_k_recall(observed, ss) == 0.5


def test_top_k_recall_empty_observed() -> None:
    """Observed vide -> recall = 0.0 (edge case)."""
    observed = ObservedLineup("C1", 2024, 5, ())
    s1 = _make_scenario(["ALPHA"], 1.0)
    ss = _make_scenario_set([s1])
    assert top_k_recall(observed, ss) == 0.0


def test_top_k_recall_gate_threshold() -> None:
    """Gate T13 >= 0.90 : 9/10 observed captured."""
    observed = _make_observed([f"P{i}" for i in range(10)])
    s1 = _make_scenario(
        [f"P{i}" for i in range(9)] + ["NOISE1"],
        1.0,
    )
    ss = _make_scenario_set([s1])
    assert top_k_recall(observed, ss) == 0.9


# --- T3b Accuracy@K ---


def test_accuracy_at_k_perfect_match() -> None:
    """Top scenario == observed -> accuracy = 1.0."""
    observed = _make_observed(["ALPHA", "BETA", "GAMMA"])
    s1 = _make_scenario(["ALPHA", "BETA", "GAMMA"], 1.0)
    ss = _make_scenario_set([s1])
    assert accuracy_at_k(observed, ss) == 1.0


def test_accuracy_at_k_partial_match() -> None:
    """Top scenario recoupe 2/3 observed -> 2/3."""
    observed = _make_observed(["ALPHA", "BETA", "GAMMA"])
    s1 = _make_scenario(["ALPHA", "BETA", "DELTA"], 1.0)
    ss = _make_scenario_set([s1])
    assert accuracy_at_k(observed, ss) == pytest.approx(2 / 3)


def test_accuracy_at_k_uses_top_weighted() -> None:
    """Top scenario = max weight, ignore others."""
    observed = _make_observed(["ALPHA", "BETA"])
    s_low = _make_scenario(["XXX", "YYY"], 0.3)
    s_top = _make_scenario(["ALPHA", "BETA"], 0.7)
    ss = _make_scenario_set([s_low, s_top])
    assert accuracy_at_k(observed, ss) == 1.0


def test_accuracy_at_k_override_k() -> None:
    """K override = consider top scenario against observed with given k."""
    observed = _make_observed(["ALPHA"])
    s1 = _make_scenario(["ALPHA", "BETA"], 1.0)
    ss = _make_scenario_set([s1])
    # observed has 1 player, top scenario contains ALPHA -> 1/1
    assert accuracy_at_k(observed, ss, k=1) == 1.0


def test_accuracy_at_k_gate_threshold() -> None:
    """Gate T13b >= 0.75 : 6/8 observed captured by top scenario."""
    observed = _make_observed([f"P{i}" for i in range(8)])
    s1 = _make_scenario(
        [f"P{i}" for i in range(6)] + ["N1", "N2"],
        1.0,
    )
    ss = _make_scenario_set([s1])
    assert accuracy_at_k(observed, ss) == 0.75


def test_accuracy_at_k_empty_scenario_set() -> None:
    """ScenarioSet vide -> accuracy = 0.0 (edge case)."""
    observed = _make_observed(["ALPHA"])
    ss = ScenarioSet(
        scenarios=(),
        opponent_club_id="C1",
        round_date="2024-01-01",
        generated_at="2024-01-01T00:00:00Z",
        lineage_hash="a" * 64,
    )
    assert accuracy_at_k(observed, ss) == 0.0


def test_accuracy_at_k_empty_observed_default_k() -> None:
    """Observed vide avec default k -> accuracy = 0.0 (k_eff=0 edge)."""
    observed = ObservedLineup("C1", 2024, 5, ())
    s1 = _make_scenario(["ALPHA"], 1.0)
    ss = _make_scenario_set([s1])
    assert accuracy_at_k(observed, ss) == 0.0
