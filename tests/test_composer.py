# tests/test_composer.py
"""
Tests ComposerService (CE) - ISO 29119

Tests unitaires pour la logique d'optimisation.
"""

import pytest
from services.composer import ComposerService


class TestComposerService:
    """Tests pour ComposerService."""

    def setup_method(self):
        """Setup avant chaque test."""
        self.service = ComposerService()

    def test_calculate_expected_score_equal_elo(self):
        """
        Test: Score attendu = 0.5 pour Elos egaux.
        """
        score = self.service.calculate_expected_score(1500, 1500)
        assert 0.45 <= score <= 0.55  # Proche de 0.5

    def test_calculate_expected_score_higher_elo_wins(self):
        """
        Test: Score attendu > 0.5 si notre Elo est superieur.
        """
        score = self.service.calculate_expected_score(1800, 1500)
        assert score > 0.5

    def test_calculate_expected_score_lower_elo_loses(self):
        """
        Test: Score attendu < 0.5 si notre Elo est inferieur.
        """
        score = self.service.calculate_expected_score(1500, 1800)
        assert score < 0.5

    def test_calculate_probabilities_sum_to_one(self):
        """
        Test: Probabilites win + draw + loss = 1.
        """
        win, draw, loss = self.service.calculate_probabilities(1600, 1500)
        total = win + draw + loss
        assert 0.99 <= total <= 1.01  # Tolerance flottants

    def test_optimize_returns_correct_team_size(self):
        """
        Test: Optimisation retourne le bon nombre de joueurs.
        """
        players = [
            {"ffe_id": f"A{i:05d}", "name": f"Player {i}", "elo": 1500 + i * 50}
            for i in range(10)
        ]
        opponents = [
            {"ffe_id": f"B{i:05d}", "elo": 1600}
            for i in range(8)
        ]

        result = self.service.optimize(
            players,
            opponents,
            {"team_size": 8, "elo_descending": True},
        )

        assert len(result.lineup) == 8

    def test_optimize_respects_elo_order(self):
        """
        Test: Joueurs tries par Elo decroissant.
        """
        players = [
            {"ffe_id": "A00001", "name": "Low", "elo": 1400},
            {"ffe_id": "A00002", "name": "High", "elo": 1800},
            {"ffe_id": "A00003", "name": "Mid", "elo": 1600},
        ]
        opponents = [{"elo": 1500} for _ in range(3)]

        result = self.service.optimize(
            players,
            opponents,
            {"team_size": 3, "elo_descending": True},
        )

        elos = [a.player_elo for a in result.lineup]
        assert elos == sorted(elos, reverse=True)

    def test_optimize_calculates_total_score(self):
        """
        Test: Score total est calcule.
        """
        players = [{"ffe_id": "A00001", "elo": 1600}]
        opponents = [{"elo": 1500}]

        result = self.service.optimize(players, opponents, {"team_size": 1})

        assert result.total_expected_score > 0
        assert result.total_expected_score == result.lineup[0].expected_score
