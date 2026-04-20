"""Tests T3 + T3b metrics (Top-K recall + Accuracy@K).

Plan 3 V2 Phase 3. ISO 29119 : coverage >= 80%, fixtures dédiées.
"""

from __future__ import annotations

import pytest

from scripts.backtest.ground_truth import ObservedLineup, ObservedPlayer
from scripts.backtest.metrics import (
    accuracy_at_k,
    brier_presence,
    brier_skill_score,
    jaccard_max,
    top_k_recall,
)
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


# --- T4 Jaccard max ---


def test_jaccard_max_perfect_match() -> None:
    """Un scenario matche exactement observed -> jaccard = 1.0."""
    observed = _make_observed(["ALPHA", "BETA", "GAMMA"])
    s1 = _make_scenario(["ALPHA", "BETA", "GAMMA"], 0.5)
    s2 = _make_scenario(["XYZ", "UVW"], 0.5)
    ss = _make_scenario_set([s1, s2])
    assert jaccard_max(observed, ss) == 1.0


def test_jaccard_max_no_intersection() -> None:
    """Aucun recoupement -> jaccard = 0.0."""
    observed = _make_observed(["ALPHA"])
    s1 = _make_scenario(["XYZ"], 1.0)
    ss = _make_scenario_set([s1])
    assert jaccard_max(observed, ss) == 0.0


def test_jaccard_max_partial() -> None:
    """|∩|=1, |∪|=3 -> jaccard = 1/3."""
    observed = _make_observed(["ALPHA", "BETA"])
    s1 = _make_scenario(["ALPHA", "GAMMA"], 1.0)
    ss = _make_scenario_set([s1])
    assert jaccard_max(observed, ss) == pytest.approx(1 / 3)


def test_jaccard_max_picks_best_across_scenarios() -> None:
    """Le max prend le meilleur scenario, pas le plus lourd."""
    observed = _make_observed(["ALPHA", "BETA"])
    s_low_match = _make_scenario(["ALPHA", "XYZ"], 0.8)  # weight=0.8 mais jac=1/3
    s_high_match = _make_scenario(["ALPHA", "BETA"], 0.2)  # weight=0.2 mais jac=1.0
    ss = _make_scenario_set([s_low_match, s_high_match])
    assert jaccard_max(observed, ss) == 1.0


def test_jaccard_max_gate_threshold() -> None:
    """Gate T14 >= 0.75 : observed et scenario different peu."""
    observed = _make_observed([f"P{i}" for i in range(8)])
    # Scenario : 7/8 match + 1 substitution -> |∩|=7, |∪|=9 -> 7/9 ≈ 0.777
    s1 = _make_scenario([f"P{i}" for i in range(7)] + ["NEW"], 1.0)
    ss = _make_scenario_set([s1])
    jac = jaccard_max(observed, ss)
    assert jac >= 0.75
    assert jac == pytest.approx(7 / 9)


# --- T5 Brier score P(presence) ---


def test_brier_presence_perfect_prediction():
    """Perfect prediction (scenario == observed) → Brier = 0."""
    observed = _make_observed(["ALPHA", "BETA"])
    s1 = _make_scenario(["ALPHA", "BETA"], 1.0)
    ss = _make_scenario_set([s1])
    assert brier_presence(observed, ss) == pytest.approx(0.0)


def test_brier_presence_all_wrong():
    """Predict XYZ weight=1, observed ALPHA → Brier = 1.0."""
    observed = _make_observed(["ALPHA"])
    s1 = _make_scenario(["XYZ"], 1.0)
    ss = _make_scenario_set([s1])
    # Union = {ALPHA X, XYZ X}. (0-1)² + (1-0)² = 2. /2 = 1.0
    assert brier_presence(observed, ss) == pytest.approx(1.0)


def test_brier_presence_weighted_half_half():
    """Weights 0.5/0.5 different scenarios → Brier = 0.25."""
    observed = _make_observed(["ALPHA"])
    s1 = _make_scenario(["ALPHA"], 0.5)
    s2 = _make_scenario(["BETA"], 0.5)
    ss = _make_scenario_set([s1, s2])
    # p_presence[ALPHA X]=0.5, p_presence[BETA X]=0.5
    # (0.5-1)²=0.25 + (0.5-0)²=0.25 = 0.5 / 2 = 0.25
    assert brier_presence(observed, ss) == pytest.approx(0.25)


def test_brier_presence_empty_observed_with_predictions():
    """Observed empty but predictions exist → Brier over predictions only."""
    observed = ObservedLineup("C1", 2024, 5, ())
    s1 = _make_scenario(["ALPHA"], 1.0)
    ss = _make_scenario_set([s1])
    # p_presence[ALPHA X]=1, obs_flag=0, (1-0)²=1 / 1 = 1.0
    assert brier_presence(observed, ss) == pytest.approx(1.0)


def test_brier_presence_gate_threshold():
    """Gate T15 ≤ 0.20 : predictions bien calibrées."""
    observed = _make_observed([f"P{i}" for i in range(4)])
    # 4 scenarios uniform weights, all covering observed fully → Brier tends to 0
    scenarios = [
        _make_scenario([f"P{i}" for i in range(4)], 0.25),
        _make_scenario([f"P{i}" for i in range(4)], 0.25),
        _make_scenario([f"P{i}" for i in range(4)], 0.25),
        _make_scenario([f"P{i}" for i in range(4)], 0.25),
    ]
    ss = _make_scenario_set(scenarios)
    assert brier_presence(observed, ss) <= 0.20


# --- T6 Brier skill score (Pappalardo 2019) ---


def test_brier_skill_score_model_equals_baseline():
    """Model brier == baseline → BSS = 0."""
    observed = _make_observed(["ALPHA"])
    s1 = _make_scenario(["XYZ"], 1.0)
    ss = _make_scenario_set([s1])
    model_b = brier_presence(observed, ss)
    assert brier_skill_score(observed, ss, baseline_brier=model_b) == pytest.approx(0.0)


def test_brier_skill_score_perfect_model():
    """Perfect model (Brier=0) → BSS = 1.0."""
    observed = _make_observed(["ALPHA"])
    s1 = _make_scenario(["ALPHA"], 1.0)
    ss = _make_scenario_set([s1])
    assert brier_skill_score(observed, ss, baseline_brier=0.5) == pytest.approx(1.0)


def test_brier_skill_score_model_better_than_baseline():
    """Model Brier=0, baseline=0.26 → BSS = 1.0."""
    observed = _make_observed(["ALPHA", "BETA"])
    s1 = _make_scenario(["ALPHA", "BETA"], 1.0)
    ss = _make_scenario_set([s1])
    assert brier_skill_score(observed, ss, baseline_brier=0.26) == pytest.approx(1.0)


def test_brier_skill_score_zero_baseline_returns_zero():
    """Division par zéro → return 0 (fallback conservateur)."""
    observed = _make_observed(["ALPHA"])
    s1 = _make_scenario(["ALPHA"], 1.0)
    ss = _make_scenario_set([s1])
    assert brier_skill_score(observed, ss, baseline_brier=0.0) == 0.0


def test_brier_skill_score_gate_threshold():
    """Gate T6 ≥ 0.05 : model Brier < baseline par ≥5%."""
    observed = _make_observed(["ALPHA", "BETA"])
    s1 = _make_scenario(["ALPHA", "BETA"], 1.0)  # perfect
    ss = _make_scenario_set([s1])
    bss = brier_skill_score(observed, ss, baseline_brier=0.30)
    assert bss >= 0.05


# --- Concern #5 : Déterminisme (ISO 29119) ---


def test_metrics_deterministic_same_input():
    """Même inputs → mêmes outputs pour toutes les metrics (pureté fonctionnelle)."""
    observed = _make_observed(["ALPHA", "BETA", "GAMMA"])
    s1 = _make_scenario(["ALPHA", "BETA"], 0.6)
    s2 = _make_scenario(["GAMMA", "DELTA"], 0.4)
    ss = _make_scenario_set([s1, s2])

    # 2 appels consécutifs → mêmes résultats bit-à-bit
    assert top_k_recall(observed, ss) == top_k_recall(observed, ss)
    assert accuracy_at_k(observed, ss) == accuracy_at_k(observed, ss)
    assert jaccard_max(observed, ss) == jaccard_max(observed, ss)
    assert brier_presence(observed, ss) == brier_presence(observed, ss)
    assert brier_skill_score(observed, ss, 0.3) == brier_skill_score(observed, ss, 0.3)


# --- Concern #6 : Edge case observed non-vide + scenarios vide ---


def test_brier_presence_observed_nonempty_scenarios_empty():
    """Observed = 2 joueurs, scenarios vide → Brier = 1.0 (aucune prédiction)."""
    observed = _make_observed(["ALPHA", "BETA"])
    empty_ss = ScenarioSet(
        scenarios=(),
        opponent_club_id="C1",
        round_date="2024-01-01",
        generated_at="2024-01-01T00:00:00Z",
        lineage_hash="a" * 64,
    )
    # p_presence empty, observed flag = 1 pour 2 joueurs
    # (0 - 1)² + (0 - 1)² = 2 / 2 = 1.0
    assert brier_presence(observed, empty_ss) == pytest.approx(1.0)


def test_top_k_recall_observed_nonempty_scenarios_empty():
    """Observed nonempty + scenarios vide → recall = 0.0."""
    observed = _make_observed(["ALPHA", "BETA"])
    empty_ss = ScenarioSet(
        scenarios=(),
        opponent_club_id="C1",
        round_date="2024-01-01",
        generated_at="2024-01-01T00:00:00Z",
        lineage_hash="a" * 64,
    )
    assert top_k_recall(observed, empty_ss) == 0.0


def test_jaccard_max_observed_nonempty_scenarios_empty():
    """Observed nonempty + scenarios vide → jaccard = 0.0."""
    observed = _make_observed(["ALPHA"])
    empty_ss = ScenarioSet(
        scenarios=(),
        opponent_club_id="C1",
        round_date="2024-01-01",
        generated_at="2024-01-01T00:00:00Z",
        lineage_hash="a" * 64,
    )
    assert jaccard_max(observed, empty_ss) == 0.0
