"""TopKEnumerator - enumere K lineups deterministes (mode dominant).

ISO 5055 : SRP, complexite <= B.
ISO 29119 : deterministe, reproductible (pas de RNG).

Algorithme :
1. Sort pool par score = taux_presence_effectif desc puis Elo desc
2. Prendre top team_size joueurs -> lineup #1 (mode dominant)
3. Pour lineups #2..K : substituer 1 joueur (le moins valuable) par un joueur
   alternatif du pool (next best non-selectionne)
4. Normaliser joint_prob en weights sommant a 1
5. Source = "topk"

Document ID: ALICE-ALI-TOPK
Version: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from services.ali.scenario import BoardAssignment, Lineup, Scenario

if TYPE_CHECKING:
    from services.ali.types import CompetitionContext, PlayerCandidate
    from services.ffe.rule_engine import RuleEngine


_DEFAULT_TAUX = 0.5  # fallback si taux_presence_effectif is None


class TopKEnumerator:
    """Enumere K lineups deterministes (mode dominant)."""

    def __init__(self, engine: RuleEngine) -> None:
        """Initialize with a RuleEngine used to filter eligible candidates."""
        self._engine = engine

    def enumerate(
        self,
        pool: list[PlayerCandidate],
        context: CompetitionContext,
        k: int,
    ) -> list[Scenario]:
        """Return K Scenario(source='topk') trie par joint_prob desc."""
        if len(pool) < context.team_size:
            raise ValueError(
                f"pool too small: {len(pool)} < team_size {context.team_size}",
            )

        # 1. Filter pool via engine (eligibility, e.g. 3.7.j elo_max)
        eligible = self._engine.filter_candidates(pool, context)
        if len(eligible) < context.team_size:
            raise ValueError(
                f"eligible pool too small: {len(eligible)} < team_size {context.team_size}",
            )

        # 2. Sort par taux_presence desc puis Elo desc (priorite presence)
        sorted_pool = sorted(
            eligible,
            key=lambda p: (
                -(p.taux_presence_effectif or _DEFAULT_TAUX),
                -p.elo,
            ),
        )

        # 3. Construire K candidats lineups via swap top <-> reserve
        lineups = self._build_k_candidates(sorted_pool, context, k)

        # 4. Build Scenarios with normalized weights (sum=1)
        scenarios = [self._build_scenario(lineup) for lineup in lineups]
        return self._normalize_weights(scenarios)

    def _build_k_candidates(
        self,
        sorted_pool: list[PlayerCandidate],
        context: CompetitionContext,
        k: int,
    ) -> list[Lineup]:
        """Genere K lineups distincts par swap top<->reserve."""
        team_size = context.team_size
        top = sorted_pool[:team_size]
        reserves = sorted_pool[team_size:]

        # Lineup #1 : top brut
        lineups = [self._make_lineup(top, team_size)]

        if k == 1:
            return lineups

        # Lineups #2..K : remplace 1 joueur du top (le moins valuable: position team_size-1)
        # par 1 reserve. Iterate sur reserves jusqu'a K total.
        for i in range(min(k - 1, len(reserves))):
            variant = top[: team_size - 1] + [reserves[i]]
            lineups.append(self._make_lineup(variant, team_size))

        # Si on n'a pas atteint k (pool insuffisant), s'arrete la
        return lineups[:k]

    @staticmethod
    def _make_lineup(players: list[PlayerCandidate], team_size: int) -> Lineup:
        """Build Lineup avec board assignment par Elo desc."""
        sorted_players = sorted(players, key=lambda p: -p.elo)
        assignments = tuple(
            BoardAssignment(
                board=i + 1,
                player=p,
                p_assignment=p.taux_presence_effectif or _DEFAULT_TAUX,
            )
            for i, p in enumerate(sorted_players[:team_size])
        )
        return Lineup(team_size=team_size, assignments=assignments)

    @staticmethod
    def _build_scenario(lineup: Lineup) -> Scenario:
        """Compute joint_prob = produit P(present) x P(board=k|present)."""
        joint = 1.0
        for a in lineup.assignments:
            joint *= a.p_assignment
        return Scenario(lineup=lineup, joint_prob=joint, weight=0.0, source="topk")

    @staticmethod
    def _normalize_weights(scenarios: list[Scenario]) -> list[Scenario]:
        """Normalize weights so sum = 1.0 across the K scenarios."""
        total = sum(s.joint_prob for s in scenarios)
        if total <= 0:
            uniform = 1.0 / len(scenarios)
            return [
                Scenario(
                    lineup=s.lineup,
                    joint_prob=s.joint_prob,
                    weight=uniform,
                    source=s.source,
                )
                for s in scenarios
            ]
        return [
            Scenario(
                lineup=s.lineup,
                joint_prob=s.joint_prob,
                weight=s.joint_prob / total,
                source=s.source,
            )
            for s in scenarios
        ]
