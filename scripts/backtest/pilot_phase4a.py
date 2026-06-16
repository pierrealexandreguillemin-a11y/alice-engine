"""Phase 4a joint-conditional local pilot — N=70 viable N3, saison 2024 (T9).

Runs the existing backtest harness in Phase 4a mode (simultaneous_teams +
target_team resolved by team name from config/clubs_teams_2024.json,
date-filtered) and emits an early-gate report: mean recall >= 0.50 decides
whether to proceed to the Kaggle Phase A run (T10). For each viable match it
also runs the Phase 3 path (no sim teams) on the SAME match for a paired
McNemar/Wilcoxon comparison. Non-viable matches are counted + logged, never
silently dropped.

Document ID: ALICE-BACKTEST-PILOT-PHASE4A
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.backtest.clubs_teams_fixture import (
    build_team_to_club_index,
    load_simultaneous_teams,
)
from scripts.backtest.ground_truth import extract_observed_lineup
from scripts.backtest.harness import BacktestHarness
from scripts.backtest.metrics import brier_presence, jaccard_max, top_k_recall
from scripts.backtest.pilot_phase4a_helpers import aggregate_stats, write_report
from scripts.backtest.runner_sampling import enumerate_candidates
from scripts.backtest.runner_types import RunnerConfig

if TYPE_CHECKING:
    from scripts.backtest.runner_types import MatchCandidate
    from services.ali.types import TeamSpec

logger = logging.getLogger(__name__)

EARLY_GATE_RECALL = 0.50

# ADR-014 validate message fragment used to classify thin-residual skips.
_THIN_RESIDUAL_MSG = "must contain 20 scenarios"


def is_thin_residual(exc: Exception) -> bool:
    """Return True when *exc* is the ADR-014 ScenarioSet validation error.

    A thin-residual skip means the joint-conditional path drained the opponent's
    superior teams leaving too few players to generate 20 distinct lineups.  It
    is an EXPECTED Phase 4a fidelity signal, NOT a code error.
    """
    return _THIN_RESIDUAL_MSG in str(exc)


MAX_VIABLE = 70  # collect this many viable (target-present + >=1-superior) matches
_FIXTURE = Path("config/clubs_teams_2024.json")


def early_gate_decision(mean_recall: float) -> str:
    """Return the early-gate verdict string (>= threshold = PASS)."""
    if mean_recall >= EARLY_GATE_RECALL:
        return (
            f"PASS (mean recall {mean_recall:.4f} >= {EARLY_GATE_RECALL}) "
            "-> proceed to Kaggle (T10)"
        )
    return (
        f"FAIL (mean recall {mean_recall:.4f} < {EARLY_GATE_RECALL}) "
        "-> STOP + diagnostic before Kaggle"
    )


def names_index(sim: list[TeamSpec], target: str) -> int:
    """Number of teams ranked strictly above `target` (= its index in `sim`)."""
    return [t.team_name for t in sim].index(target)


def _viable_sim(
    payload: dict[str, Any], team_index: dict[str, str], cand: MatchCandidate
) -> list[TeamSpec] | None:
    """Ordered sim list IF the match is viable, else None.

    Viable = the target opponent team is present in the date-filtered fixture
    list AND has at least one superior team (it is not team_1). A missing date
    cannot be resolved against the fixture -> non-viable.
    """
    if not cand.date:
        return None
    sim = load_simultaneous_teams(
        payload,
        team_name=cand.opp_team,
        ronde=cand.ronde,
        match_date=cand.date,
        team_index=team_index,
    )
    names = [t.team_name for t in sim]
    if cand.opp_team not in names:  # club absent OR target dropped by date filter
        return None
    if names.index(cand.opp_team) == 0:  # target is team_1 -> no superior team
        return None
    return sim


def _config() -> RunnerConfig:
    """N3 saison-2024 config covering all rounds (enumerate_candidates only)."""
    return RunnerConfig(
        saison=2024,
        rondes=tuple(range(1, 12)),
        max_matches=10_000,
        team_size=8,
        division="N3",
        division_filter="Nationale 3",
        type_competition="national",
        nb_rondes_total=11,
        seed=42,
    )


def _run_pair(
    harness: BacktestHarness,
    cfg: RunnerConfig,
    cand: MatchCandidate,
    sim: list[TeamSpec],
) -> tuple[Any, Any, Any]:
    """Run Phase 4a + paired Phase 3 + extract observed. Returns the 3 results."""
    user_club_id = harness.cache.team_to_club[cand.user_team]
    common = {
        "user_club_id": user_club_id,
        "opponent_club_id": cand.opp_club,
        "saison": cand.saison,
        "ronde": cand.ronde,
        "nb_rondes_total": cfg.nb_rondes_total,
        "division": cfg.division,
        "team_size": cfg.team_size,
        "user_lineup": [],
        "seed": cfg.seed,
        "strict": False,
    }
    result = harness.run_match(
        **common, round_date=cand.date, simultaneous_teams=sim, target_team=cand.opp_team
    )
    baseline = harness.run_match(**common)
    observed = extract_observed_lineup(
        harness.cache,
        cand.opp_team,
        cand.saison,
        cand.ronde,
        as_domicile=False,
        groupe=cand.groupe,
    )
    return result, baseline, observed


def _row(
    cand: MatchCandidate, sim: list[TeamSpec], result: Any, baseline: Any, observed: Any
) -> dict[str, Any]:
    """Build one per-match result row from the predicted + observed lineups."""
    return {
        "opp_team": cand.opp_team,
        "ronde": cand.ronde,
        "date": cand.date,
        "n_superior": names_index(sim, cand.opp_team),
        "recall": top_k_recall(observed, result.scenario_set),
        "recall_baseline": top_k_recall(observed, baseline.scenario_set),
        "jaccard": jaccard_max(observed, result.scenario_set),
        "brier": brier_presence(observed, result.scenario_set),
    }


def run_pilot(out_dir: Path = Path("reports")) -> dict[str, Any]:
    """Run the Phase 4a pilot and write reports/pilot_phase4a.md."""
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    team_index = build_team_to_club_index(payload)
    harness = BacktestHarness()
    harness.setup()
    cfg = _config()
    candidates = enumerate_candidates(harness.cache, cfg)

    rows: list[dict[str, Any]] = []
    skipped = {"non_viable": 0, "no_observed": 0, "thin_residual": 0, "error": 0}
    for cand in candidates:
        if len(rows) >= MAX_VIABLE:
            break
        sim = _viable_sim(payload, team_index, cand)
        if sim is None:
            skipped["non_viable"] += 1
            continue
        try:
            result, baseline, observed = _run_pair(harness, cfg, cand, sim)
        except ValueError as exc:
            if is_thin_residual(exc):
                logger.info(
                    "pilot match skipped (thin residual pool): ronde=%s opp=%s date=%s",
                    cand.ronde,
                    cand.opp_team,
                    cand.date,
                )
                skipped["thin_residual"] += 1
                continue
            logger.exception(
                "pilot match failed: ronde=%s opp=%s date=%s club=%s",
                cand.ronde,
                cand.opp_team,
                cand.date,
                cand.opp_club,
            )
            skipped["error"] += 1
            continue
        except Exception:  # noqa: BLE001  intentional: one bad match must not kill the pilot
            logger.exception(
                "pilot match failed: ronde=%s opp=%s date=%s club=%s",
                cand.ronde,
                cand.opp_team,
                cand.date,
                cand.opp_club,
            )
            skipped["error"] += 1
            continue
        if not observed.players:
            skipped["no_observed"] += 1
            continue
        rows.append(_row(cand, sim, result, baseline, observed))

    n = len(rows)
    mean_recall = sum(r["recall"] for r in rows) / n if n else 0.0
    mean_baseline = sum(r["recall_baseline"] for r in rows) / n if n else 0.0
    summary: dict[str, Any] = {
        "n_matches": n,
        "max_viable": MAX_VIABLE,
        "mean_recall": mean_recall,
        "mean_baseline_recall": mean_baseline,
        "skipped": skipped,
        "decision": early_gate_decision(mean_recall),
        **aggregate_stats(rows),
    }
    write_report(out_dir, rows, summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print(run_pilot()["decision"])
