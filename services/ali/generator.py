"""ScenarioGenerator - orchestrateur ALI Plan 2.

ISO 5259 : lineage_hash propage bout-en-bout.
ISO 5055 : SRP, complexite <= B.
ISO 29119 : deterministe via seed.
ISO 25059 : T18-T21 enforced via ScenarioSet.validate() + dedup distincts.

Document ID: ALICE-ALI-GENERATOR
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from services.ali.generator_helpers import merge_and_pad, normalize_history, renormalize
from services.ali.joint_conditional import compute_adverse_exclusions
from services.ali.joint_sampler import CopulaJointSampler
from services.ali.monte_carlo import MonteCarloSampler
from services.ali.scenario import ScenarioSet
from services.ali.topk import TopKEnumerator
from services.ffe.rule_engine import RuleEngine

if TYPE_CHECKING:
    from services.ali.cache import ALIDataCache
    from services.ali.history import HistoryEnricher
    from services.ali.pool_loader import PlayerPoolLoader
    from services.ali.types import CompetitionContext, PlayerCandidate, TeamSpec
    from services.ali.verifiability import VerifiabilityClassifier


class ScenarioGenerator:
    """Orchestrateur ALI : pool -> enrich -> fit copula -> TopK + MC -> ScenarioSet."""

    def __init__(  # noqa: PLR0913
        self,
        engine: RuleEngine,
        classifier: VerifiabilityClassifier,
        cache: ALIDataCache,
        pool_loader: PlayerPoolLoader,
        history_enricher: HistoryEnricher,
        decay_lambda: float = 0.9,
    ) -> None:
        """Store collaborators; no I/O here."""
        self._engine = engine
        self._classifier = classifier
        self._cache = cache
        self._pool_loader = pool_loader
        self._history_enricher = history_enricher
        self._decay_lambda = decay_lambda

    def generate(  # noqa: PLR0913
        self,
        opponent_club_id: str,
        round_date: str,
        context: CompetitionContext,
        saison: int,
        current_round: int,
        nb_rondes_total: int,
        overrides: list[dict[str, Any]] | None = None,
        n_topk: int = 10,
        n_mc_pairs: int = 5,
        seed: int = 42,
        simultaneous_teams: list[TeamSpec] | None = None,
        target_team: str | None = None,
    ) -> ScenarioSet:
        """Generate a ScenarioSet (10 TopK + 10 MC = 20 scenarios).

        Phase 3 (`simultaneous_teams is None`): sample the target club's full
        pool. Phase 4a (`simultaneous_teams` provided): exclude players consumed
        by the club's superior teams (A02 §3.7.b top-down) via the CE-adverse
        mirror, then run the same Phase 3 pipeline over the residual pool
        (D-P3-19 fix).
        """
        exclude_players: set[str] = set()
        if simultaneous_teams is not None:
            if target_team is None:
                raise ValueError("target_team required when simultaneous_teams is set")
            full_pool = self._pool_loader.load_pool(opponent_club_id, round_date, overrides)
            exclude_players = compute_adverse_exclusions(
                pool=full_pool,
                teams=simultaneous_teams,
                target_team=target_team,
                seed=seed,
            )

        # 1. Load pool (F7 implicit via membership; Phase 4a drains superior teams)
        pool = self._pool_loader.load_pool(
            opponent_club_id, round_date, overrides, exclude_players=exclude_players
        )
        if len(pool) < context.team_size:
            raise ValueError(
                f"pool too small for {opponent_club_id}: {len(pool)} < {context.team_size}",
            )
        return self._run_phase3(
            pool,
            context=context,
            saison=saison,
            current_round=current_round,
            nb_rondes_total=nb_rondes_total,
            n_topk=n_topk,
            n_mc_pairs=n_mc_pairs,
            seed=seed,
            opponent_club_id=opponent_club_id,
            round_date=round_date,
            target_team=target_team,
            simultaneous_teams=simultaneous_teams,
        )

    def _run_phase3(  # noqa: PLR0913
        self,
        pool: list[PlayerCandidate],
        *,
        context: CompetitionContext,
        saison: int,
        current_round: int,
        nb_rondes_total: int,
        n_topk: int,
        n_mc_pairs: int,
        seed: int,
        opponent_club_id: str,
        round_date: str,
        target_team: str | None = None,
        simultaneous_teams: list[TeamSpec] | None = None,
    ) -> ScenarioSet:
        """Phase 3 pipeline over a (possibly reduced) pool: enrich -> copula -> TopK + MC."""
        # 2. Enrich features (F2 recency + F3 streak)
        enriched = self._history_enricher.enrich(
            pool,
            saison=saison,
            current_round=current_round,
            nb_rondes_total=nb_rondes_total,
        )

        # 3. Fit copula on history for this club's players
        copula = self._fit_copula(
            enriched,
            saison=saison,
            current_round=current_round,
            nb_rondes_total=nb_rondes_total,
        )

        # 4. Partition rules PUBLIC / PRIVATE (spec §4.7 step 5)
        public_rules, _private_rules = self._classifier.partition_rules(self._engine.rules)
        public_engine = RuleEngine(
            rules=list(public_rules),
            source_sha256=f"{self._engine.lineage_hash()}+public",
        )

        # 5. TopK enumeration (public rules uniquement, spec step 6)
        topk_enum = TopKEnumerator(engine=public_engine)
        topk_scenarios = topk_enum.enumerate(enriched, context, k=n_topk)

        # 6. MC sampling (LHS + antithetic + copula, public rules uniquement, spec step 7)
        mc = MonteCarloSampler(engine=public_engine, copula=copula)
        rng = np.random.default_rng(seed)
        mc_scenarios = mc.sample(enriched, context, n_pairs=n_mc_pairs, rng=rng)

        # 7. Merge + dedup distinct (T20)
        merged = merge_and_pad(
            list(topk_scenarios),
            list(mc_scenarios),
            mc,
            enriched,
            context,
            rng,
        )

        # 8. Renormalize weights so sum = 1.0
        merged = renormalize(merged)

        # 9. Build ScenarioSet with lineage_hash
        lineage = self._compute_lineage(
            opponent_club_id,
            round_date,
            context,
            saison,
            current_round,
            nb_rondes_total,
            n_topk,
            n_mc_pairs,
            seed,
            target_team,
            simultaneous_teams,
        )
        result = ScenarioSet(
            scenarios=tuple(merged),
            opponent_club_id=opponent_club_id,
            round_date=round_date,
            generated_at=datetime.now(UTC).isoformat(),
            lineage_hash=lineage,
        )
        result.validate()  # T18, T19
        return result

    def _fit_copula(
        self,
        enriched: list[PlayerCandidate],
        saison: int,
        current_round: int,
        nb_rondes_total: int,
    ) -> CopulaJointSampler:
        """Fit copula on history for enriched pool."""
        copula = CopulaJointSampler(decay_lambda=self._decay_lambda)
        names = [f"{c.nom} {c.prenom}".strip() for c in enriched]
        history = self._cache.lookup_history(names)
        history_normalized = normalize_history(history)
        copula.fit(
            history=history_normalized,
            player_names=names,
            saison=saison,
            nb_rondes_total=nb_rondes_total,
            current_round=current_round,
        )
        return copula

    def _compute_lineage(  # noqa: PLR0913
        self,
        opp_id: str,
        round_date: str,
        ctx: CompetitionContext,
        saison: int,
        cur_round: int,
        nb_rondes: int,
        n_topk: int,
        n_mc_pairs: int,
        seed: int,
        target_team: str | None = None,
        simultaneous_teams: list[TeamSpec] | None = None,
    ) -> str:
        """SHA-256 of all input parameters + cache + rules signatures (ISO 5259).

        Phase 4a inputs (target_team + simultaneous_teams) are folded in so two
        calls differing only in which team is the target produce DISTINCT hashes
        (cache-key correctness, cf. D-2026-06-16-phase-4a-part-2b-api-cache). When
        both are None (Phase 3 path) no extra bytes are added -> legacy hash preserved.
        """
        m = hashlib.sha256()
        m.update(f"opp={opp_id}|date={round_date}|niveau={ctx.niveau}|".encode())
        m.update(
            f"team_size={ctx.team_size}|saison={saison}|cur={cur_round}|".encode(),
        )
        m.update(
            f"nb_rondes={nb_rondes}|n_topk={n_topk}|n_mc={n_mc_pairs}|seed={seed}|".encode(),
        )
        m.update(f"lambda={self._decay_lambda}|".encode())
        m.update(f"rules_sig={self._engine.lineage_hash()}|".encode())
        m.update(f"j_sig={self._cache.parquet_sig_joueurs}|".encode())
        m.update(f"e_sig={self._cache.parquet_sig_echiquiers}|".encode())
        if target_team is not None:
            m.update(f"target={target_team}|".encode())
        if simultaneous_teams:
            teams_sig = "|".join(f"{t.team_name}:{t.board_count}" for t in simultaneous_teams)
            m.update(f"sim_teams={teams_sig}|".encode())
        return m.hexdigest()
