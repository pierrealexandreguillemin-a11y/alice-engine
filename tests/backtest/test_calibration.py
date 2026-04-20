"""Tests T7 ECE + T7b Reliability diagram — Plan 3 V2 Phase 3.

ISO 29119 : unit tests synthétiques, gate threshold coverage, edge cases.
"""

from __future__ import annotations

import pytest

from scripts.backtest.calibration import (
    ReliabilityPoint,
    ece_presence,
    reliability_diagram,
)
from scripts.backtest.ground_truth import ObservedLineup, ObservedPlayer
from services.ali.scenario import BoardAssignment, Lineup, Scenario, ScenarioSet
from services.ali.types import PlayerCandidate


def _mk_player(nr: str, nom: str, prenom: str, elo: int = 1800) -> PlayerCandidate:
    return PlayerCandidate(
        nr_ffe=nr,
        nom=nom,
        prenom=prenom,
        elo=elo,
        club="C1",
        mute=False,
        genre="M",
        categorie="SE",
        licence_active=True,
    )


def _mk_scenario(players: list[PlayerCandidate], weight: float, source: str = "topk") -> Scenario:
    assigns = tuple(
        BoardAssignment(board=i + 1, player=p, p_assignment=1.0) for i, p in enumerate(players)
    )
    return Scenario(
        lineup=Lineup(team_size=len(players), assignments=assigns),
        joint_prob=weight,
        weight=weight,
        source=source,
    )


def _mk_scenario_set(scenarios: tuple[Scenario, ...]) -> ScenarioSet:
    return ScenarioSet(
        scenarios=scenarios,
        opponent_club_id="C1",
        round_date="2024-01-01",
        generated_at="2024-01-01T00:00:00Z",
        lineage_hash="test" + "0" * 60,
    )


def _mk_observed(names: list[str]) -> ObservedLineup:
    players = tuple(
        ObservedPlayer(joueur_nom=n, echiquier=i + 1, elo=1800) for i, n in enumerate(names)
    )
    return ObservedLineup(club_name="C1", saison=2024, ronde=5, players=players)


# ============ T7 ECE ============


def test_ece_perfect_calibration_returns_zero():
    """Tous les joueurs prédits à p=1.0 ET observés : ECE = 0."""
    pA = _mk_player("1", "DUPONT", "Jean")
    pB = _mk_player("2", "MARTIN", "Paul")
    ss = _mk_scenario_set((_mk_scenario([pA, pB], 1.0),))
    obs = _mk_observed(["DUPONT Jean", "MARTIN Paul"])
    assert ece_presence(obs, ss, n_bins=10) == pytest.approx(0.0)


def test_ece_total_mismatch_high():
    """Prédits p=1.0 mais jamais observés : ECE = 1.0 (max)."""
    pA = _mk_player("1", "DUPONT", "Jean")
    ss = _mk_scenario_set((_mk_scenario([pA], 1.0),))
    obs = _mk_observed([])
    # 1 item dans bin [0.9, 1.0] : conf=1.0, acc=0.0, ECE=1.0
    assert ece_presence(obs, ss, n_bins=10) == pytest.approx(1.0)


def test_ece_half_half_weighted():
    """2 scenarios poids 0.5 chacun, 1 joueur partagé : p=1.0, obs=1 → ECE=0."""
    pA = _mk_player("1", "DUPONT", "Jean")
    s1 = _mk_scenario([pA], 0.5)
    s2 = _mk_scenario([pA], 0.5)
    ss = _mk_scenario_set((s1, s2))
    obs = _mk_observed(["DUPONT Jean"])
    assert ece_presence(obs, ss) == pytest.approx(0.0)


def test_ece_empty_returns_zero():
    """Union vide → ECE = 0."""
    ss = _mk_scenario_set((_mk_scenario([], 1.0),))
    obs = _mk_observed([])
    assert ece_presence(obs, ss) == pytest.approx(0.0)


def test_ece_gate_threshold():
    """ECE <= 0.05 → well-calibrated (Plan 3 V2 P3G10 strict)."""
    pA = _mk_player("1", "DUPONT", "Jean")
    pB = _mk_player("2", "MARTIN", "Paul")
    # 2 joueurs, 1 observé, 1 non → ECE = 0.5 (fail gate)
    ss = _mk_scenario_set((_mk_scenario([pA, pB], 1.0),))
    obs = _mk_observed(["DUPONT Jean"])
    ece = ece_presence(obs, ss)
    assert ece > 0.05  # fail gate


def test_ece_invalid_n_bins_raises():
    pA = _mk_player("1", "DUPONT", "Jean")
    ss = _mk_scenario_set((_mk_scenario([pA], 1.0),))
    obs = _mk_observed(["DUPONT Jean"])
    with pytest.raises(ValueError, match="n_bins"):
        ece_presence(obs, ss, n_bins=0)


def test_ece_deterministic_same_input():
    pA = _mk_player("1", "DUPONT", "Jean")
    pB = _mk_player("2", "MARTIN", "Paul")
    ss = _mk_scenario_set((_mk_scenario([pA, pB], 1.0),))
    obs = _mk_observed(["DUPONT Jean"])
    assert ece_presence(obs, ss) == ece_presence(obs, ss)


def test_ece_single_bin_reduces_to_abs_diff():
    """n_bins=1 → ECE = |mean_p - mean_obs| sur tous les items."""
    pA = _mk_player("1", "A", "a")
    pB = _mk_player("2", "B", "b")
    # p=0.5 each (weight sums), obs = 1 for A, 0 for B → ECE = |0.5 - 0.5| = 0
    s1 = _mk_scenario([pA, pB], 0.5)
    ss = _mk_scenario_set((s1,))
    obs = _mk_observed(["A a"])
    # After weight aggregation: p(A) = 0.5, p(B) = 0.5
    # Bin [0, 1]: mean_p = 0.5, mean_obs = 0.5 → ECE = 0
    ece = ece_presence(obs, ss, n_bins=1)
    assert ece == pytest.approx(0.0)


# ============ T7b Reliability ============


def test_reliability_diagram_returns_points():
    pA = _mk_player("1", "DUPONT", "Jean")
    ss = _mk_scenario_set((_mk_scenario([pA], 1.0),))
    obs = _mk_observed(["DUPONT Jean"])
    pts = reliability_diagram(obs, ss, n_bins=10)
    assert len(pts) == 1
    pt = pts[0]
    assert isinstance(pt, ReliabilityPoint)
    assert pt.mean_predicted == pytest.approx(1.0)
    assert pt.observed_frequency == pytest.approx(1.0)
    assert pt.count == 1


def test_reliability_diagram_empty_returns_empty():
    ss = _mk_scenario_set((_mk_scenario([], 1.0),))
    obs = _mk_observed([])
    assert reliability_diagram(obs, ss) == []


def test_reliability_diagram_bins_ordered():
    pA = _mk_player("1", "A", "a")
    pB = _mk_player("2", "B", "b")
    s1 = _mk_scenario([pA], 1.0)  # p(A) = 1.0 bin [0.9, 1.0]
    s2 = _mk_scenario([pB], 0.3)  # p(B) = 0.3 bin [0.3, 0.4]
    ss = ScenarioSet(
        scenarios=(s1, s2),
        opponent_club_id="C1",
        round_date="2024-01-01",
        generated_at="2024-01-01T00:00:00Z",
        lineage_hash="test" + "0" * 60,
    )
    obs = _mk_observed(["A a"])
    pts = reliability_diagram(obs, ss, n_bins=10)
    # 2 bins non vides ordered
    assert len(pts) == 2
    assert pts[0].bin_low < pts[1].bin_low


def test_reliability_diagram_point_is_frozen():
    """ReliabilityPoint immutable (ISO 29119)."""
    pt = ReliabilityPoint(
        bin_low=0.0, bin_high=0.1, mean_predicted=0.05, observed_frequency=0.0, count=3
    )
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        pt.bin_low = 0.5  # type: ignore[misc]


def test_reliability_diagram_invalid_n_bins_raises():
    pA = _mk_player("1", "A", "a")
    ss = _mk_scenario_set((_mk_scenario([pA], 1.0),))
    obs = _mk_observed(["A a"])
    with pytest.raises(ValueError, match="n_bins"):
        reliability_diagram(obs, ss, n_bins=-1)
