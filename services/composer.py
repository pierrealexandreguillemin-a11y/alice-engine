# services/composer.py
"""
CE - Composition Engine

Service d'optimisation de la composition utilisateur.
Logique metier pure, sans I/O direct (SRP).

@description Optimise la composition pour maximiser le score attendu
@see CDC_ALICE.md - F.2 Optimisation de Composition
@see ISO 42010 - Service layer
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BoardAssignment:
    """Assignation d'un joueur a un echiquier."""

    board: int
    player_ffe_id: str
    player_name: str
    player_elo: int
    opponent_ffe_id: str | None
    opponent_elo: int | None
    expected_score: float
    win_probability: float
    draw_probability: float
    loss_probability: float


@dataclass
class CompositionResult:
    """Resultat de l'optimisation."""

    lineup: list[BoardAssignment]
    total_expected_score: float
    confidence: float


class ComposerService:
    """
    Service CE (Composition Engine).

    Responsabilite: Optimiser la composition utilisateur
    pour maximiser le score attendu contre l'adversaire predit.

    Methodes:
        - optimize: Trouve la composition optimale
        - calculate_expected_score: Calcule le score attendu pour un matchup
        - get_alternatives: Genere des compositions alternatives
    """

    # Constantes pour le calcul Elo
    ELO_K_FACTOR = 400  # Facteur K standard
    DRAW_PROBABILITY_FACTOR = 0.35  # Probabilite moyenne de nulle

    def optimize(
        self,
        available_players: list[dict[str, Any]],
        predicted_opponents: list[dict[str, Any]],
        constraints: dict[str, Any] | None = None,
    ) -> CompositionResult:
        """
        Optimise la composition pour maximiser le score attendu.

        @param available_players: Joueurs disponibles avec Elo
        @param predicted_opponents: Adversaires predits par echiquier
        @param constraints: Contraintes (ordre Elo, etc.)

        @returns: Composition optimale avec score attendu

        @see ISO 25010 - Performance (optimisation)
        """
        if constraints is None:
            constraints = {"elo_descending": True, "team_size": 8}

        team_size = constraints.get("team_size", 8)
        elo_descending = constraints.get("elo_descending", True)

        # Trier les joueurs disponibles par Elo
        sorted_players = sorted(
            available_players,
            key=lambda p: p.get("elo", 0),
            reverse=True,
        )

        # Si ordre Elo obligatoire, prendre les N meilleurs
        if elo_descending:
            selected = sorted_players[:team_size]
        else:
            # TODO: Utiliser OR-Tools pour optimisation sans contrainte Elo
            selected = sorted_players[:team_size]

        # Calculer les assignations
        lineup = []
        total_score = 0.0

        for i, player in enumerate(selected):
            board = i + 1
            opponent = predicted_opponents[i] if i < len(predicted_opponents) else None

            opponent_elo = opponent.get("elo", 1500) if opponent else 1500
            player_elo = player.get("elo", 1500)

            # Calculer probabilites
            win_prob, draw_prob, loss_prob = self.calculate_probabilities(player_elo, opponent_elo)
            expected = win_prob + (draw_prob * 0.5)

            assignment = BoardAssignment(
                board=board,
                player_ffe_id=player.get("ffe_id", ""),
                player_name=player.get("name", ""),
                player_elo=player_elo,
                opponent_ffe_id=opponent.get("ffe_id") if opponent else None,
                opponent_elo=opponent_elo if opponent else None,
                expected_score=expected,
                win_probability=win_prob,
                draw_probability=draw_prob,
                loss_probability=loss_prob,
            )

            lineup.append(assignment)
            total_score += expected

        return CompositionResult(
            lineup=lineup,
            total_expected_score=total_score,
            confidence=0.5,  # TODO: Calculer vraie confiance
        )

    def calculate_probabilities(
        self,
        player_elo: int,
        opponent_elo: int,
    ) -> tuple[float, float, float]:
        """
        Calcule les probabilites victoire/nulle/defaite via formule Elo.

        @param player_elo: Elo du joueur
        @param opponent_elo: Elo de l'adversaire

        @returns: Tuple (win_prob, draw_prob, loss_prob)

        @see Formule Elo standard
        """
        # Formule Elo: P(win) = 1 / (1 + 10^((Elo_adv - Elo_mon)/400))
        diff = opponent_elo - player_elo
        expected = 1 / (1 + math.pow(10, diff / self.ELO_K_FACTOR))

        # Ajustement pour les nulles (modele simplifie)
        # Plus les Elos sont proches, plus la nulle est probable
        elo_diff_abs = abs(diff)
        draw_factor = max(0.1, self.DRAW_PROBABILITY_FACTOR - (elo_diff_abs / 2000))

        # Repartition: expected = win + 0.5*draw
        # Donc: win = expected - 0.5*draw
        win_prob = max(0, expected - (draw_factor * 0.5))
        draw_prob = draw_factor
        loss_prob = max(0, 1 - win_prob - draw_prob)

        # Normaliser pour que la somme = 1
        total = win_prob + draw_prob + loss_prob
        if total > 0:
            win_prob /= total
            draw_prob /= total
            loss_prob /= total

        return (win_prob, draw_prob, loss_prob)

    def calculate_expected_score(
        self,
        player_elo: int,
        opponent_elo: int,
    ) -> float:
        """
        Calcule le score attendu (0-1) pour un matchup.

        @see CDC_ALICE.md - Formule Elo
        """
        win, draw, _ = self.calculate_probabilities(player_elo, opponent_elo)
        return win + (draw * 0.5)

    def get_alternatives(
        self,
        available_players: list[dict[str, Any]],
        predicted_opponents: list[dict[str, Any]],
        constraints: dict[str, Any] | None = None,
        count: int = 3,
    ) -> list[CompositionResult]:
        """
        Genere des compositions alternatives.

        @param count: Nombre d'alternatives a generer

        @returns: Liste de compositions alternatives
        """
        # TODO: Implementer avec OR-Tools
        # - Variante defensive (minimiser pertes)
        # - Variante agressive (maximiser gains)
        # - Variante robuste (minimiser variance)
        return []
