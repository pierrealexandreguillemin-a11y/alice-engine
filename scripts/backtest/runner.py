"""T11 Runner full walk-forward backtest (Plan 3 V2 Phase 3).

Orchestrateur qui assemble tous les modules T1-T10 pour produire un
rapport de quality gates P3G sur hold-out 2024.

Pipeline par match :
    1. sample (saison, ronde, equipe_ext) depuis cache.echiquiers
    2. resolve equipe_ext -> club via cache.team_to_club (H1 fix)
    3. harness.run_match(opponent=club) -> ScenarioSet ALI
    4. extract_observed_lineup(equipe_ext) -> ObservedLineup
    5. compute 5 metrics ALI + 5 metrics baseline Elo
    6. per-match "correct" flag pour McNemar paired test
    7. aggregate : bootstrap BCa CI + McNemar + summary

ISO 25059 : quality gates. ISO 29119 : backtest strict walk-forward.
ISO 42001 : lineage_hash par match + report JSON.

Sources SOTA
------------
- Bergmeir & Benítez 2012 "On the use of cross-validation for time series
  predictor evaluation" (Info Sciences 191) - walk-forward protocol.
- Pappalardo et al. 2019, PlayeRank - paired comparison baseline.
- Efron 1987 "Better Bootstrap Confidence Intervals" (JASA 82) - BCa.

Document ID: ALICE-BACKTEST-RUNNER
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scripts.backtest.baseline_elo import baseline_elo_scenario_set
from scripts.backtest.bootstrap import bootstrap_ci
from scripts.backtest.calibration import ece_presence
from scripts.backtest.ground_truth import extract_observed_lineup
from scripts.backtest.metrics import (
    accuracy_at_k,
    brier_presence,
    brier_skill_score,
    jaccard_max,
    top_k_recall,
)
from scripts.backtest.runner_types import (
    RECALL_GATE,
    BacktestReport,
    MatchStats,
    RunnerConfig,
    df_to_candidates,
)
from scripts.backtest.statistical import mcnemar_paired

if TYPE_CHECKING:
    from scripts.backtest.harness import BacktestHarness

logger = logging.getLogger(__name__)


@dataclass
class BacktestRunner:
    """Walk-forward backtest orchestrator (T11 Plan 3 V2).

    Usage :
        harness = BacktestHarness()
        harness.setup()
        runner = BacktestRunner(harness=harness, config=RunnerConfig())
        report = runner.run()
        print(report.gates_summary())
    """

    harness: BacktestHarness
    config: RunnerConfig = field(default_factory=RunnerConfig)

    def sample_matches(self) -> list[tuple[int, int, str, str, str]]:
        """Sample hold-out matches (saison, ronde, user_team, opp_team, opp_club).

        @returns list up to ``max_matches`` quintuples. Dedupe par
                 (user_team, opp_team) pour eviter redondance.
        """
        if self.harness.cache is None:
            msg = "Harness not setup()"
            raise RuntimeError(msg)
        df = self.harness.cache.echiquiers_total
        team_to_club = self.harness.cache.team_to_club
        joueurs_by_club = self.harness.cache.joueurs_by_club

        matches: list[tuple[int, int, str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for ronde in self.config.rondes:
            sub = df[(df["saison"] == self.config.saison) & (df["ronde"] == ronde)]
            sub = sub.drop_duplicates(subset=["equipe_dom", "equipe_ext"])
            for _, row in sub.iterrows():
                user_team = str(row["equipe_dom"])
                opp_team = str(row["equipe_ext"])
                key = (user_team, opp_team)
                if key in seen:
                    continue
                opp_club = team_to_club.get(opp_team)
                user_club = team_to_club.get(user_team)
                if opp_club is None or user_club is None:
                    continue
                user_pool = joueurs_by_club.get(user_club)
                opp_pool = joueurs_by_club.get(opp_club)
                if user_pool is None or opp_pool is None:
                    continue
                if len(user_pool) < self.config.team_size or len(opp_pool) < self.config.team_size:
                    continue
                matches.append((self.config.saison, ronde, user_team, opp_team, opp_club))
                seen.add(key)
                if len(matches) >= self.config.max_matches:
                    return matches
        return matches

    def run_single(  # noqa: PLR0913, PLR0914
        self,
        saison: int,
        ronde: int,
        user_team: str,
        opp_team: str,
        opp_club: str,
    ) -> MatchStats | None:
        """Run one match : ALI predictions + baseline + metrics."""
        cache = self.harness.cache
        if cache is None:
            msg = "Harness cache None"
            raise RuntimeError(msg)
        user_club = cache.team_to_club.get(user_team)
        if user_club is None:
            return None
        user_pool = cache.joueurs_by_club[user_club]
        user_lineup = [
            {"ffe_id": str(p["nr_ffe"]), "elo": int(p["elo"] or 1500)}
            for _, p in user_pool.head(self.config.team_size).iterrows()
        ]

        try:
            result = self.harness.run_match(
                user_club_id=user_club,
                opponent_club_id=opp_club,
                saison=saison,
                ronde=ronde,
                nb_rondes_total=self.config.nb_rondes_total,
                division=self.config.division,
                team_size=self.config.team_size,
                user_lineup=user_lineup,
                seed=self.config.seed,
                strict=False,
            )
            observed = extract_observed_lineup(cache, opp_team, saison, ronde, as_domicile=False)
        except Exception:
            logger.exception("match failed saison=%s ronde=%s opp=%s", saison, ronde, opp_team)
            if self.config.skip_failed_matches:
                return None
            raise

        if len(observed.players) == 0:
            return None

        baseline_pool = df_to_candidates(cache.joueurs_by_club[opp_club], opp_club)
        if len(baseline_pool) < self.config.team_size:
            return None
        baseline_ss = baseline_elo_scenario_set(baseline_pool, self.config.team_size)

        recall_ali = top_k_recall(observed, result.scenario_set)
        recall_base = top_k_recall(observed, baseline_ss)
        brier_base = brier_presence(observed, baseline_ss)

        return MatchStats(
            saison=saison,
            ronde=ronde,
            user_team=user_team,
            opponent_team=opp_team,
            recall_ali=recall_ali,
            accuracy_ali=accuracy_at_k(observed, result.scenario_set),
            jaccard_ali=jaccard_max(observed, result.scenario_set),
            brier_ali=brier_presence(observed, result.scenario_set),
            ece_ali=ece_presence(observed, result.scenario_set),
            recall_baseline=recall_base,
            brier_baseline=brier_base,
            bss=brier_skill_score(observed, result.scenario_set, brier_base),
            ali_correct=recall_ali >= RECALL_GATE,
            baseline_correct=recall_base >= RECALL_GATE,
        )

    def run(self) -> BacktestReport:
        """Execute walk-forward backtest + aggregate report.

        @raises ValueError: fewer than 2 matches completed (bootstrap needs N>=2).
        """
        stats: list[MatchStats] = []
        for saison, ronde, user_team, opp_team, opp_club in self.sample_matches():
            s = self.run_single(saison, ronde, user_team, opp_team, opp_club)
            if s is not None:
                stats.append(s)
        if len(stats) < 2:
            msg = f"Need >= 2 completed matches for bootstrap, got {len(stats)}"
            raise ValueError(msg)

        def ci(vals: list[float]) -> object:
            return bootstrap_ci(
                vals,
                confidence=self.config.bootstrap_confidence,
                n_resamples=self.config.n_bootstrap,
                seed=self.config.seed,
            )

        return BacktestReport(
            n_matches=len(stats),
            per_match=stats,
            ci_recall=ci([s.recall_ali for s in stats]),  # type: ignore[arg-type]
            ci_accuracy=ci([s.accuracy_ali for s in stats]),  # type: ignore[arg-type]
            ci_jaccard=ci([s.jaccard_ali for s in stats]),  # type: ignore[arg-type]
            ci_brier=ci([s.brier_ali for s in stats]),  # type: ignore[arg-type]
            ci_ece=ci([s.ece_ali for s in stats]),  # type: ignore[arg-type]
            mean_bss=sum(s.bss for s in stats) / len(stats),
            mcnemar=mcnemar_paired(
                [s.ali_correct for s in stats],
                [s.baseline_correct for s in stats],
            ),
        )
