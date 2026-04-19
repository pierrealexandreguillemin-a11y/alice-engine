"""TopKEnumerator - enumere K lineups deterministes via branch-and-bound priorise.

ISO 5055 : SRP, complexite <= B.
ISO 29119 : deterministe, reproductible (pas de RNG, priority queue a tie-break stable).

Algorithme (spec Phase 3 Plan 2 §4.6) :
1. Pre-filter pool via `engine.filter_candidates` (eligibility, e.g. 3.7.j).
2. Sort par `(P(present, recency+streak) DESC, Elo DESC)`.
3. Initial lineup #1 : top `team_size` joueurs (mode dominant).
4. Branch-and-bound priorise : priority queue (heapq) sur `-joint_prob`.
   - Extraction DESC par joint_prob.
   - Neighbors via swap (position x reserve) sur lineup courant.
   - Validate via `engine.validate_lineup` (prune si violation severite "error").
   - Dedup via signature canonique (nr_ffe:board).
5. Retourner K lineups distincts, normaliser weights (sum=1), source="topk".

Complexite : O(K x team_size x |reserves|) expansions, heap ops O(log n).
Garantie DESC stricte de joint_prob (ordre d'extraction heap).

Document ID: ALICE-ALI-TOPK
Version: 2.0.0
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from services.ali.scenario import BoardAssignment, Lineup, Scenario
from services.ali.types import CompetitionContext, PlayerCandidate

if TYPE_CHECKING:
    from services.ffe.rule_engine import RuleEngine


_DEFAULT_TAUX = 0.5  # fallback si taux_presence_effectif is None


# Heap entry : (-joint_prob, tie, lineup, current_top_players).
_HeapEntry = tuple[float, int, Lineup, tuple[PlayerCandidate, ...]]


@dataclass
class _SearchState:
    """Mutable state of branch-and-bound search (heap + dedup + tie counter).

    Regroupe les structures partagees par _expand_neighbors et
    _try_push_variant pour respecter ISO 5055 (PLR0913).
    """

    reserves: list[PlayerCandidate]
    context: CompetitionContext
    baseline_articles: frozenset[str]
    seen: set[tuple[str, ...]] = field(default_factory=set)
    heap: list[_HeapEntry] = field(default_factory=list)
    tie_counter: int = 0


class TopKEnumerator:
    """Enumere K lineups deterministes via branch-and-bound priorise (spec §4.6)."""

    def __init__(self, engine: RuleEngine) -> None:
        """Initialize with a RuleEngine used to filter/validate candidates."""
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

        # 1. Filter pool via engine (eligibility)
        eligible = self._engine.filter_candidates(pool, context)
        if len(eligible) < context.team_size:
            raise ValueError(
                f"eligible pool too small: {len(eligible)} < team_size {context.team_size}",
            )

        # 2. Sort par taux_presence desc puis Elo desc
        sorted_pool = sorted(
            eligible,
            key=lambda p: (
                -(p.taux_presence_effectif or _DEFAULT_TAUX),
                -p.elo,
            ),
        )

        # 3. Branch-and-bound priorise pour K lineups distincts
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
        """Branch-and-bound priorise pour top-K lineups distincts par joint_prob.

        Priority queue (heapq min-heap sur -joint_prob) explore neighbors
        par ordre decroissant de joint_prob, avec deduplication via signature
        canonique et pruning via RuleEngine.validate_lineup.

        Baseline : on accepte comme valide tout lineup qui ne viole PAS
        de regles supplementaires par rapport au lineup initial (top brut).
        Si une regle est deja violee par le meilleur lineup possible (ex.
        foreign_quota / fr_gender manquant dans le pool), on ne peut pas
        esperer la satisfaire par swap : on la traite comme contrainte
        structurelle inchangeable et on ne prune PAS sur elle.
        """
        team_size = context.team_size
        top = sorted_pool[:team_size]
        reserves = sorted_pool[team_size:]

        initial_lineup = self._make_lineup(top, team_size)
        results: list[Lineup] = [initial_lineup]

        if k == 1 or not reserves:
            return results[:k]

        # Baseline violations : regles deja violees par le top brut.
        # Un neighbor est rejete seulement s'il introduit de NOUVELLES
        # violations (ex. mutes passe de 2 a 4 avec max_mutes=3).
        state = _SearchState(
            reserves=reserves,
            context=context,
            baseline_articles=self._violated_articles(top, context),
            seen={self._signature(initial_lineup)},
        )
        self._expand_neighbors(top, state)

        while state.heap and len(results) < k:
            _neg_prob, _tie, lineup, current_top = heapq.heappop(state.heap)
            sig = self._signature(lineup)
            if sig in state.seen:
                continue
            state.seen.add(sig)
            results.append(lineup)
            self._expand_neighbors(list(current_top), state)

        return results

    def _violated_articles(
        self,
        players: list[PlayerCandidate],
        context: CompetitionContext,
    ) -> frozenset[str]:
        """Return articles violes (severity error) par le lineup players."""
        return frozenset(
            v.rule_article
            for v in self._engine.validate_lineup(players, context)
            if v.severity == "error"
        )

    def _expand_neighbors(
        self,
        current_top: list[PlayerCandidate],
        state: _SearchState,
    ) -> None:
        """Generate neighbors via swap (position x reserve), push valid into heap.

        Muter `state` (heap, seen, tie_counter). Pruning via
        RuleEngine.validate_lineup sauf articles deja dans
        state.baseline_articles (structurellement non satisfaisables).
        """
        current_ids = {p.nr_ffe for p in current_top}
        for pos_idx in range(len(current_top)):
            for reserve in state.reserves:
                if reserve.nr_ffe in current_ids:
                    continue
                self._try_push_variant(current_top, pos_idx, reserve, state)

    def _try_push_variant(
        self,
        current_top: list[PlayerCandidate],
        pos_idx: int,
        reserve: PlayerCandidate,
        state: _SearchState,
    ) -> None:
        """Build a variant (swap pos_idx with reserve), validate, push if OK."""
        variant_players = list(current_top)
        variant_players[pos_idx] = reserve
        variant_lineup = self._make_lineup(variant_players, state.context.team_size)
        sig = self._signature(variant_lineup)
        if sig in state.seen:
            return
        new_violations = self._violated_articles(variant_players, state.context)
        # Prune only if variant INTRODUIT de nouvelles violations
        # (articles non presents dans baseline = structurellement respectables).
        if new_violations - state.baseline_articles:
            return
        prob = self._joint_prob(variant_lineup)
        heapq.heappush(
            state.heap,
            (-prob, state.tie_counter, variant_lineup, tuple(variant_players)),
        )
        state.tie_counter += 1

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
    def _signature(lineup: Lineup) -> tuple[str, ...]:
        """Canonical signature : tuple des (nr_ffe:board) pour T20 dedup."""
        return tuple(
            f"{a.player.nr_ffe}:{a.board}"
            for a in sorted(lineup.assignments, key=lambda a: a.board)
        )

    @staticmethod
    def _joint_prob(lineup: Lineup) -> float:
        """Compute joint_prob = produit des p_assignment."""
        p = 1.0
        for a in lineup.assignments:
            p *= a.p_assignment
        return p

    @staticmethod
    def _build_scenario(lineup: Lineup) -> Scenario:
        """Wrap Lineup dans Scenario (unnormalized weight)."""
        joint = TopKEnumerator._joint_prob(lineup)
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
