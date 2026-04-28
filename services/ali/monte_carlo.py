"""MonteCarloSampler - LHS + antithetic + copula inverse-transform (F5 SOTA).

ISO 42001 : SOTA documented (McKay 1979, Hammersley & Morton 1956, Owen 2013).
ISO 25010 : rejection_rate observable via last_rejection_rate property.
ISO 5055 : SRP, complexite <= B (helpers extraits).

Algorithme :
1. LHS sample u in [0,1]^N x n_pairs (scipy.stats.qmc.LatinHypercube)
2. Antithetic pairs : pour chaque u, ajouter (1-u) -> 2*n_pairs scenarios
3. Per scenario : inverse-transform via copula -> presence binary
4. Select team_size + assign boards Elo desc
5. RuleEngine.validate_lineup -> reject/resample max 50 retries
6. Return Scenarios source='monte_carlo' avec weights normalized

References :
- McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison of
  three methods for selecting values of input variables in the analysis
  of output from a computer code. Technometrics 21(2), 239-245.
- Hammersley, J. M. & Morton, K. W. (1956). A new Monte Carlo technique:
  antithetic variates. Math. Proc. Cambridge Phil. Soc. 52(3), 449-475.
- Owen, A. B. (2013). Monte Carlo theory, methods and examples.

Document ID: ALICE-ALI-MONTE-CARLO
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import qmc

from services.ali.scenario import BoardAssignment, Lineup, Scenario

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from services.ali.joint_sampler import CopulaJointSampler
    from services.ali.types import CompetitionContext, PlayerCandidate
    from services.ffe.rule_engine import RuleEngine


_DEFAULT_TAUX = 0.5
_MAX_RETRIES = 50
logger = logging.getLogger(__name__)


class MonteCarloSampler:
    """Monte Carlo sampler with LHS + antithetic via Gaussian copula."""

    def __init__(self, engine: RuleEngine, copula: CopulaJointSampler) -> None:
        """Initialize with a RuleEngine and a fitted CopulaJointSampler."""
        self._engine = engine
        self._copula = copula
        self._last_rejection_rate: float = 0.0

    @property
    def last_rejection_rate(self) -> float:
        """Fraction of MC attempts rejected by RuleEngine (ISO 25010 observability)."""
        return self._last_rejection_rate

    def sample(
        self,
        pool: list[PlayerCandidate],
        context: CompetitionContext,
        n_pairs: int,
        rng: np.random.Generator,
    ) -> list[Scenario]:
        """Return 2 * n_pairs Scenarios (LHS + antithetic) source='monte_carlo'."""
        if n_pairs == 0:
            self._last_rejection_rate = 0.0
            return []
        if len(pool) < context.team_size:
            raise ValueError(
                f"pool too small: {len(pool)} < team_size {context.team_size}",
            )

        n_samples = 2 * n_pairs
        n_dim = len(pool)

        # 1. LHS + antithetic uniform samples
        u_samples = self._lhs_antithetic_samples(n_pairs, n_dim, rng)

        # 2. Per sample : copula inverse-transform -> presence -> lineup
        scenarios: list[Scenario] = []
        rejected = 0
        for i in range(n_samples):
            scenario = self._sample_one_with_retry(
                u_samples[i],
                pool,
                context,
                rng,
            )
            if scenario is None:
                rejected += 1
                continue
            scenarios.append(scenario)

        total_attempts = n_samples + rejected
        self._last_rejection_rate = rejected / total_attempts if total_attempts > 0 else 0.0

        # 3. Pad if short (rejected too many): re-sample without LHS
        while len(scenarios) < n_samples:
            extra = self._sample_one_with_retry(
                rng.uniform(size=n_dim),
                pool,
                context,
                rng,
            )
            if extra is not None:
                scenarios.append(extra)
            else:
                logger.warning(
                    "MC could not fill %d/%d scenarios",
                    n_samples,
                    len(scenarios),
                )
                break

        return self._normalize_weights(scenarios[:n_samples])

    @staticmethod
    def _lhs_antithetic_samples(
        n_pairs: int,
        n_dim: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Return shape (2 * n_pairs, n_dim) array of uniform samples.

        Half from LHS, other half antithetic (1 - u).
        """
        sampler = qmc.LatinHypercube(d=n_dim, seed=rng)
        u = sampler.random(n=n_pairs)  # shape (n_pairs, n_dim)
        antithetic = 1.0 - u
        stacked: NDArray[np.float64] = np.vstack([u, antithetic])
        return stacked

    def _sample_one_with_retry(
        self,
        u_sample: NDArray[np.float64],
        pool: list[PlayerCandidate],
        context: CompetitionContext,
        rng: np.random.Generator,
    ) -> Scenario | None:
        """Single MC scenario : copula inverse-transform + lineup build + validate.

        Returns None if rejected after _MAX_RETRIES.
        """
        for attempt in range(_MAX_RETRIES):
            # First attempt uses provided u_sample, retries draw fresh uniform
            u = u_sample if attempt == 0 else rng.uniform(size=len(pool))
            presence = self._u_to_presence(u, pool)
            lineup = self._build_lineup(presence, pool, context)
            if lineup is None:
                continue
            players = [a.player for a in lineup.assignments]
            violations = self._engine.validate_lineup(players, context)
            hard_errors = [v for v in violations if v.severity == "error"]
            if not hard_errors:
                return self._build_scenario(lineup)
        return None

    def _u_to_presence(
        self,
        u: NDArray[np.float64],
        pool: list[PlayerCandidate],
    ) -> NDArray[np.int8]:
        """Inverse-transform via copule fittee (structure correlation preservee).

        Fail-fast (D-P3-12 resolved 2026-04-28) : si la copula n'est pas fittee
        ou si n_players mismatch, raise plutot que fallback threshold marginal
        independant. Le fallback indep produirait des samples non-correles
        (effet substitution joueur n°1/n°9 perdu) -> biais conservateur ML
        silencieux. Mieux vaut crash que biais cache (ISO 24029 fail-fast).

        Le `ScenarioGenerator` (Plan 2 Task 7) garantit toujours une copula
        fittee via `_fit_copula` avant l'appel. Cette branche raise est donc
        defensive : elle protege contre un futur wiring incorrect.
        """
        if not self._copula.is_fit:
            msg = (
                "MonteCarloSampler._u_to_presence: copula not fit. "
                "Generator must call CopulaJointSampler.fit(...) before sample(). "
                "ISO 24029 fail-fast (no silent fallback to independent threshold)."
            )
            raise RuntimeError(msg)
        if self._copula.n_players != len(pool):
            msg = (
                f"MonteCarloSampler._u_to_presence: copula n_players "
                f"{self._copula.n_players} != pool size {len(pool)}. "
                f"Refit copula on the same pool before sampling."
            )
            raise RuntimeError(msg)
        return self._copula.transform_uniform_to_presence(u)

    @staticmethod
    def _build_lineup(
        presence: NDArray[np.int8],
        pool: list[PlayerCandidate],
        context: CompetitionContext,
    ) -> Lineup | None:
        """Build lineup from presence vector. Returns None if not enough present."""
        present_indices = np.where(presence == 1)[0]
        if len(present_indices) < context.team_size:
            return None
        present_players = [pool[i] for i in present_indices]
        # Sort by Elo desc + take top team_size
        sorted_players = sorted(present_players, key=lambda p: -p.elo)[: context.team_size]
        assignments = tuple(
            BoardAssignment(
                board=i + 1,
                player=p,
                p_assignment=p.taux_presence_effectif or _DEFAULT_TAUX,
            )
            for i, p in enumerate(sorted_players)
        )
        return Lineup(team_size=context.team_size, assignments=assignments)

    @staticmethod
    def _build_scenario(lineup: Lineup) -> Scenario:
        """Compute joint_prob = product of per-board p_assignment."""
        joint = 1.0
        for a in lineup.assignments:
            joint *= a.p_assignment
        return Scenario(lineup=lineup, joint_prob=joint, weight=0.0, source="monte_carlo")

    @staticmethod
    def _normalize_weights(scenarios: list[Scenario]) -> list[Scenario]:
        """Normalize weights so sum = 1.0 across the scenarios."""
        if not scenarios:
            return []
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
