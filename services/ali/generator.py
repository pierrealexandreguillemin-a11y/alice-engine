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
import pandas as pd

from services.ali.joint_sampler import CopulaJointSampler
from services.ali.monte_carlo import MonteCarloSampler
from services.ali.scenario import Scenario, ScenarioSet
from services.ali.topk import TopKEnumerator
from services.ffe.rule_engine import RuleEngine

if TYPE_CHECKING:
    from services.ali.cache import ALIDataCache
    from services.ali.history import HistoryEnricher
    from services.ali.pool_loader import PlayerPoolLoader
    from services.ali.types import CompetitionContext, PlayerCandidate
    from services.ali.verifiability import VerifiabilityClassifier


_EXPECTED_TOTAL = 20


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
    ) -> ScenarioSet:
        """Generate ScenarioSet (10 TopK + 10 MC = 20 scenarios)."""
        # 1. Load pool (F7 implicit via membership)
        pool = self._pool_loader.load_pool(opponent_club_id, round_date, overrides)
        if len(pool) < context.team_size:
            raise ValueError(
                f"pool too small for {opponent_club_id}: {len(pool)} < {context.team_size}",
            )

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
        merged = self._merge_and_pad(
            list(topk_scenarios),
            list(mc_scenarios),
            mc,
            enriched,
            context,
            rng,
        )

        # 8. Renormalize weights so sum = 1.0
        merged = self._renormalize(merged)

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
        history_normalized = self._normalize_history(history)
        copula.fit(
            history=history_normalized,
            player_names=names,
            saison=saison,
            nb_rondes_total=nb_rondes_total,
            current_round=current_round,
        )
        return copula

    def _merge_and_pad(  # noqa: PLR0913
        self,
        topk: list[Scenario],
        mc: list[Scenario],
        mc_sampler: MonteCarloSampler,
        pool: list[PlayerCandidate],
        context: CompetitionContext,
        rng: np.random.Generator,
    ) -> list[Scenario]:
        """Dedup + pad to _EXPECTED_TOTAL (retries up to 5 rounds best-effort)."""
        merged = self._dedup_distinct(topk + mc)
        existing_sigs = {_signature(s) for s in merged}
        max_rounds = 5
        for _ in range(max_rounds):
            missing = _EXPECTED_TOTAL - len(merged)
            if missing <= 0:
                break
            extras = self._pad_with_mc(mc_sampler, pool, context, missing, rng)
            progressed = False
            for e in extras:
                sig = _signature(e)
                if sig not in existing_sigs:
                    merged.append(e)
                    existing_sigs.add(sig)
                    progressed = True
                if len(merged) >= _EXPECTED_TOTAL:
                    break
            if not progressed:
                break  # pool trop petit pour generer 20 lineups distincts
        return merged[:_EXPECTED_TOTAL]

    @staticmethod
    def _dedup_distinct(scenarios: list[Scenario]) -> list[Scenario]:
        """Garde scenarios distincts par signature (player nr_ffe x board)."""
        seen: set[tuple[tuple[str, int], ...]] = set()
        out: list[Scenario] = []
        for s in scenarios:
            sig = _signature(s)
            if sig not in seen:
                seen.add(sig)
                out.append(s)
        return out

    @staticmethod
    def _pad_with_mc(
        mc: MonteCarloSampler,
        pool: list[PlayerCandidate],
        context: CompetitionContext,
        n_extra: int,
        rng: np.random.Generator,
    ) -> list[Scenario]:
        """Best-effort pad with extra MC samples if dedup left us short.

        Double the request to increase diversity (LHS + antithetic produces
        many duplicates when the dominant lineup has very high joint_prob).
        """
        if n_extra <= 0:
            return []
        n_pairs = max(2, n_extra)  # oversample to improve diversity
        extra = mc.sample(pool, context, n_pairs=n_pairs, rng=rng)
        return list(extra)

    @staticmethod
    def _renormalize(scenarios: list[Scenario]) -> list[Scenario]:
        """Normalize weights so sum = 1.0 across the final set."""
        total = sum(s.joint_prob for s in scenarios)
        if total <= 0:
            uniform = 1.0 / len(scenarios) if scenarios else 0.0
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

    @staticmethod
    def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
        """Echiquiers raw -> joueur_nom long format (reuse pattern history.py)."""
        parts: list[pd.DataFrame] = []
        for col in ("blanc_nom", "noir_nom"):
            if col in df.columns:
                sub = df[[col, "saison", "ronde", "echiquier"]].copy()
                sub = sub.rename(columns={col: "joueur_nom"})
                parts.append(sub)
        if not parts:
            return pd.DataFrame(
                columns=["joueur_nom", "saison", "ronde", "echiquier"],
            )
        out = pd.concat(parts, ignore_index=True)
        return out.drop_duplicates(subset=["joueur_nom", "saison", "ronde"])

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
    ) -> str:
        """SHA-256 of all input parameters + cache + rules signatures (ISO 5259)."""
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
        return m.hexdigest()


def _signature(s: Scenario) -> tuple[tuple[str, int], ...]:
    """Canonical signature (nr_ffe x board) for dedup."""
    return tuple((a.player.nr_ffe, a.board) for a in s.lineup.assignments)
