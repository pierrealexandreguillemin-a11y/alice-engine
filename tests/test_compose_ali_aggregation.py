"""Unit tests for D-P2-06 fix : scenario aggregation in routes.py.

Spec 2026-04-19-phase3-ali-monte-carlo-design.md §2 §4.7
E[score_board_k] = Σ_s weight_s × E[score_board_k|scenario_s]

ISO 29119 : structured tests + independent fixtures.
ISO 42001 : determinism + traceability of aggregation weights.

Document ID: ALICE-TEST-D-P2-06
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from services.ali.aggregation import (
    ScenarioAggregationCtx,
    aggregate_from_scenarios,
    aggregate_one_board,
)
from services.ali.scenario import BoardAssignment, Lineup, Scenario, ScenarioSet
from services.ali.types import PlayerCandidate
from services.inference import PredictionResult


def _mk_player(nr_ffe: str, elo: int) -> PlayerCandidate:
    """Build a minimal PlayerCandidate for tests."""
    return PlayerCandidate(
        nr_ffe=nr_ffe,
        nom="X",
        prenom="Y",
        elo=elo,
        club="C",
        mute=False,
        genre="M",
        categorie="Sen",
        licence_active=True,
    )


def _mock_feature_store():
    """Mock FeatureStore returning dummy single-row DataFrame."""
    import pandas as pd

    mock = MagicMock()
    mock.assemble.return_value = pd.DataFrame(
        [{"blanc_elo": 1500, "noir_elo": 1500, "diff_elo": 0}]
    )
    return mock


def _mk_scenario(players: list[tuple[str, int]], weight: float, source: str = "topk") -> Scenario:
    """Build a Scenario with given (nr_ffe, elo) board assignments."""
    assignments = tuple(
        BoardAssignment(board=i + 1, player=_mk_player(pid, elo), p_assignment=1.0)
        for i, (pid, elo) in enumerate(players)
    )
    lineup = Lineup(team_size=len(players), assignments=assignments)
    return Scenario(lineup=lineup, joint_prob=weight, weight=weight, source=source)


def _mk_scenario_set(scenarios: list[Scenario]) -> ScenarioSet:
    """Build a ScenarioSet (validate()-skipping variant for unit tests)."""
    return ScenarioSet(
        scenarios=tuple(scenarios),
        opponent_club_id="OPP1",
        round_date="2026-01-01",
        generated_at="2026-01-01T00:00:00Z",
        lineage_hash="a" * 64,
    )


@dataclass
class _StubResult:
    """Stub that the mock inference returns."""

    p_win: float
    p_draw: float
    p_loss: float

    @property
    def e_score(self) -> float:
        return self.p_win + 0.5 * self.p_draw


def _make_inference(return_by_opp_elo: dict[int, _StubResult]) -> MagicMock:
    """Build a mock StackingInferenceService that returns different probas per opp_elo."""
    svc = MagicMock()

    def _predict(player_elo, opponent_elo, features, draw_rate_lookup=None):
        r = return_by_opp_elo[opponent_elo]
        return PredictionResult(
            p_loss=r.p_loss,
            p_draw=r.p_draw,
            p_win=r.p_win,
            e_score=r.e_score,
        )

    svc.predict_board = MagicMock(side_effect=_predict)
    return svc


class TestAggregateOneBoard:
    """Unit tests for aggregate_one_board (per-board weighted avg)."""

    def test_single_scenario_weight_1_equals_raw_prediction(self):
        """1 scenario with weight=1 -> aggregated = raw prediction."""
        scenarios = [_mk_scenario([("A00001", 1500)], weight=1.0)]
        scenario_set = _mk_scenario_set(scenarios)
        user_player = {"ffe_id": "U00001", "elo": 1500}

        inference = _make_inference({1500: _StubResult(0.50, 0.20, 0.30)})

        ctx = ScenarioAggregationCtx(
            scenario_set=scenario_set,
            user_lineup=[user_player],
            team_size=1,
            ronde=1,
            division="N3",
        )
        agg = aggregate_one_board(
            inference=inference,
            feature_store=_mock_feature_store(),
            user_player=user_player,
            board_idx=0,
            ctx=ctx,
        )

        assert agg.p_win == pytest.approx(0.50)
        assert agg.p_draw == pytest.approx(0.20)
        assert agg.p_loss == pytest.approx(0.30)
        assert agg.e_score == pytest.approx(0.60)
        assert agg.mode_opponent_ffe == "A00001"
        assert agg.mean_opponent_elo == 1500

    def test_two_scenarios_weighted_average(self):
        """2 scenarios with weights 0.7/0.3 -> weighted avg of probas."""
        s1 = _mk_scenario([("A11111", 1800)], weight=0.7)
        s2 = _mk_scenario([("B22222", 1400)], weight=0.3)
        scenario_set = _mk_scenario_set([s1, s2])
        user_player = {"ffe_id": "U00001", "elo": 1500}

        # Distinct prediction per opponent_elo so we can check weighted avg
        inference = _make_inference(
            {
                1800: _StubResult(p_win=0.20, p_draw=0.30, p_loss=0.50),
                1400: _StubResult(p_win=0.70, p_draw=0.20, p_loss=0.10),
            }
        )

        ctx = ScenarioAggregationCtx(
            scenario_set=scenario_set,
            user_lineup=[user_player],
            team_size=1,
            ronde=1,
            division="N3",
        )
        agg = aggregate_one_board(
            inference=inference,
            feature_store=_mock_feature_store(),
            user_player=user_player,
            board_idx=0,
            ctx=ctx,
        )

        # Expected : p_win = 0.7*0.20 + 0.3*0.70 = 0.35
        #           p_draw = 0.7*0.30 + 0.3*0.20 = 0.27
        #           p_loss = 0.7*0.50 + 0.3*0.10 = 0.38
        assert agg.p_win == pytest.approx(0.35, abs=1e-4)
        assert agg.p_draw == pytest.approx(0.27, abs=1e-4)
        assert agg.p_loss == pytest.approx(0.38, abs=1e-4)
        # e_score = p_win + 0.5*p_draw = 0.35 + 0.135 = 0.485
        assert agg.e_score == pytest.approx(0.485, abs=1e-4)
        # Sum = 1
        assert agg.p_win + agg.p_draw + agg.p_loss == pytest.approx(1.0, abs=1e-4)
        # Mode opponent = A11111 (weight 0.7 > B 0.3)
        assert agg.mode_opponent_ffe == "A11111"
        # Weighted avg elo : 0.7*1800 + 0.3*1400 = 1680
        assert agg.mean_opponent_elo == 1680

    def test_three_scenarios_same_opponent_weights_accumulate(self):
        """3 scenarios with same opponent on the board -> weights accumulate for mode."""
        s1 = _mk_scenario([("A11111", 1600)], weight=0.3)
        s2 = _mk_scenario([("A11111", 1600)], weight=0.3)
        s3 = _mk_scenario([("B22222", 1600)], weight=0.4)
        scenario_set = _mk_scenario_set([s1, s2, s3])
        user_player = {"ffe_id": "U00001", "elo": 1500}

        inference = _make_inference({1600: _StubResult(0.40, 0.20, 0.40)})

        ctx = ScenarioAggregationCtx(
            scenario_set=scenario_set,
            user_lineup=[user_player],
            team_size=1,
            ronde=1,
            division="N3",
        )
        agg = aggregate_one_board(
            inference=inference,
            feature_store=_mock_feature_store(),
            user_player=user_player,
            board_idx=0,
            ctx=ctx,
        )

        # A11111 has accumulated weight 0.6 > B22222's 0.4 -> mode = A11111
        assert agg.mode_opponent_ffe == "A11111"


class TestAggregateFromScenarios:
    """Unit tests for aggregate_from_scenarios (full scenario-set aggregation)."""

    def test_n_boards_matches_team_size_and_user_lineup(self):
        """Returns exactly min(team_size, len(user_lineup)) BoardResults."""
        # 2 scenarios, team_size=2
        board_ass = [("A11111", 1700), ("C33333", 1600)]
        board_ass2 = [("B22222", 1800), ("D44444", 1500)]
        s1 = _mk_scenario(board_ass, weight=0.6)
        s2 = _mk_scenario(board_ass2, weight=0.4)
        scenario_set = _mk_scenario_set([s1, s2])

        user_lineup = [
            {"ffe_id": "U00001", "elo": 1500},
            {"ffe_id": "U00002", "elo": 1400},
        ]
        inference = _make_inference(
            {
                1700: _StubResult(0.40, 0.20, 0.40),
                1800: _StubResult(0.30, 0.20, 0.50),
                1600: _StubResult(0.50, 0.20, 0.30),
                1500: _StubResult(0.60, 0.15, 0.25),
            }
        )

        ctx = ScenarioAggregationCtx(
            scenario_set=scenario_set,
            user_lineup=user_lineup,
            team_size=2,
            ronde=1,
            division="N3",
        )
        aggregated = aggregate_from_scenarios(inference, _mock_feature_store(), ctx)

        assert len(aggregated) == 2
        assert aggregated[0].board == 1
        assert aggregated[1].board == 2

    def test_all_boards_probas_sum_to_one(self):
        """Each board's aggregated probas sum to ~1."""
        scenarios = [
            _mk_scenario([("A11111", 1800), ("B22222", 1600)], weight=0.5),
            _mk_scenario([("C33333", 1700), ("D44444", 1500)], weight=0.5),
        ]
        scenario_set = _mk_scenario_set(scenarios)
        user_lineup = [
            {"ffe_id": "U00001", "elo": 1500},
            {"ffe_id": "U00002", "elo": 1400},
        ]
        inference = _make_inference(
            {
                1800: _StubResult(0.20, 0.20, 0.60),
                1700: _StubResult(0.30, 0.25, 0.45),
                1600: _StubResult(0.45, 0.20, 0.35),
                1500: _StubResult(0.55, 0.15, 0.30),
            }
        )

        ctx = ScenarioAggregationCtx(
            scenario_set=scenario_set,
            user_lineup=user_lineup,
            team_size=2,
            ronde=1,
            division="N3",
        )
        aggregated = aggregate_from_scenarios(inference, _mock_feature_store(), ctx)

        for b in aggregated:
            assert b.p_win + b.p_draw + b.p_loss == pytest.approx(1.0, abs=1e-4)
            assert 0.0 <= b.e_score <= 1.0

    def test_inference_called_n_scenarios_times_n_boards(self):
        """D-P2-06 core assertion : predict_board called 20 x K times (not just K)."""
        n_scenarios = 5
        n_boards = 2
        # Build n_scenarios scenarios each with n_boards
        scenarios = []
        for i in range(n_scenarios):
            boards_def = [(f"A{i:05d}", 1500 + i * 10) for _ in range(n_boards)]
            scenarios.append(_mk_scenario(boards_def, weight=1.0 / n_scenarios))
        scenario_set = _mk_scenario_set(scenarios)
        user_lineup = [{"ffe_id": f"U{k:05d}", "elo": 1500} for k in range(n_boards)]

        elo_map = {1500 + i * 10: _StubResult(0.40, 0.20, 0.40) for i in range(n_scenarios)}
        inference = _make_inference(elo_map)

        ctx = ScenarioAggregationCtx(
            scenario_set=scenario_set,
            user_lineup=user_lineup,
            team_size=n_boards,
            ronde=1,
            division="N3",
        )
        aggregate_from_scenarios(inference, _mock_feature_store(), ctx)

        # predict_board called n_scenarios * n_boards times (vs 1 old fix = n_boards)
        assert inference.predict_board.call_count == n_scenarios * n_boards
