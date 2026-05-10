"""D8 per-saison orchestrator — kernel entry point ALICE_SAISON.

Pipeline spec §7.2 : lineage → loader → BacktestRunner → breakdowns 7 dims
→ multicalibration → conformal split 90% → stress/DRO summary → JSON.
Stress + DRO perturbation closures deferred to D-2026-05-09-d8-perturbation-runs.
Output schema d8.v1 §7.3 via D8SaisonReport (scripts/d8/types.py).
ISO 27034 (saison range), 5259 (SHA-256 lineage), 5055 (<300 lines).

Document ID: ALICE-D8-RUN
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Kaggle entry point pattern : setup sys.path BEFORE imports
if "/kaggle/input/notebooks" in os.environ.get("PWD", ""):
    sys.path.insert(0, "/kaggle/input/notebooks/pierrax/alice-d8-code")

from scripts.d8 import breakdowns, conformal, loader
from scripts.d8.types import D8GroupBreakdown, D8Lineage, D8SaisonReport

# Perturbation modules deferred until cache-mutation infra ready (Task 14).
# Importing here as documented integration points for Task 11 E2E + Task 14 :
#   from scripts.d8 import calibration, dro, stress_elo, stress_roster  # noqa: ERA001

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SAISON_MIN, SAISON_MAX = 2018, 2030
NOISE_LEVELS_ELO: tuple[float, ...] = (0.01, 0.03, 0.05, 0.07, 0.10)
TURNOVER_RATES_ROSTER: tuple[float, ...] = (0.05, 0.10, 0.20)
DRO_EPSILONS: tuple[float, ...] = (0.05, 0.10)
DRO_N_PERTURBATIONS = 50
CONFORMAL_CALIB_N = 30
CONFORMAL_ALPHA = 0.10  # 1 - 0.90 coverage


def _validate_saison(saison: int) -> None:
    """ISO 27034 input validation on saison."""
    if not SAISON_MIN <= saison <= SAISON_MAX:
        msg = f"ALICE_SAISON {saison} outside [{SAISON_MIN}, {SAISON_MAX}]"
        raise ValueError(msg)


def _input_paths() -> dict[str, Path]:
    """Resolve Kaggle dataset input paths."""
    base = Path("/kaggle/input/datasets/pierrax/alice-d8-input")
    return {
        "joueurs": base / "data/joueurs.parquet",
        "echiquiers": base / "data/echiquiers.parquet",
        "mlp": base / "artefacts/mlp_meta_learner.joblib",
        "temp_scaler": base / "artefacts/temp_scaler.joblib",
    }


def _build_lineage(saison: int, paths: dict[str, Path]) -> D8Lineage:
    """ISO 5259 SHA-256 lineage at kernel start."""
    return D8Lineage(
        joueurs_sha256=loader.compute_file_sha256(paths["joueurs"]),
        echiquiers_sha256=loader.compute_file_sha256(paths["echiquiers"]),
        mlp_artefact_sha256=loader.compute_file_sha256(paths["mlp"]),
        temp_scaler_sha256=loader.compute_file_sha256(paths["temp_scaler"]),
        code_sha256=os.environ.get("ALICE_CODE_SHA", "unknown"),
        ali_seed=42,
        ali_n_topk=10,
        ali_n_mc_pairs=5,
        ali_decay_lambda=0.9,
        kernel_id=f"pierrax/d8-saison-{saison}",
        kernel_version_kaggle=os.environ.get("KAGGLE_KERNEL_VERSION", "v1"),
        run_at_utc=datetime.now(UTC).isoformat(),
    )


def _run_backtest(saison: int) -> Any:
    """Boot Plan 3 BacktestHarness + run one saison via BacktestRunner."""
    from scripts.backtest.harness import BacktestHarness
    from scripts.backtest.runner import BacktestRunner
    from scripts.backtest.runner_sampling import RunnerConfig

    harness = BacktestHarness()
    harness.setup()
    config = RunnerConfig(saisons=(saison,))
    runner = BacktestRunner(harness=harness, config=config)
    return runner, runner.run()


def _serialize_breakdowns(
    bdowns: dict[str, D8GroupBreakdown],
) -> dict[str, dict[str, Any]]:
    """Convert D8GroupBreakdown dict to JSON-serializable form."""
    return {
        dim: {
            "dim_name": bd.dim_name,
            "groups": {gname: asdict(gs) for gname, gs in bd.groups.items()},
        }
        for dim, bd in bdowns.items()
    }


def _flatten_per_match(per_match: list[Any]) -> list[dict[str, Any]]:
    """Convert MatchStats dataclass list to dict list (asdict-safe)."""
    return [asdict(m) if is_dataclass(m) else dict(m.__dict__) for m in per_match]


def _compute_breakdowns_stage(
    runner: Any,
    per_match: list[Any],
) -> dict[str, D8GroupBreakdown]:
    """Step 4 : breakdowns 7 dims. Categorical fallbacks per ISO 24027 §6.1."""
    cache = runner.harness.cache
    if cache is None:
        msg = "Harness cache None — setup() not called"
        raise RuntimeError(msg)

    def _pool_size(m: Any) -> int:
        opp_club = cache.team_to_club.get(m.opponent_team)
        if opp_club is None:
            return 0
        return int(len(cache.joueurs_by_club.get(opp_club, [])))

    def _team_elo_mean(m: Any) -> int:
        user_club = cache.team_to_club.get(m.user_team)
        if user_club is None:
            return 1500
        pool = cache.joueurs_by_club.get(user_club)
        if pool is None or len(pool) == 0:
            return 1500
        elos = [int(p.get("elo") or 1500) for _, p in pool.head(8).iterrows()]
        return int(sum(elos) / len(elos))

    all_pool_sizes = [_pool_size(m) for m in per_match]
    return breakdowns.compute_all_7(
        per_match,
        gender_fn=lambda _m: "mixed",  # Interclubs Open default; Phase 4+ J02 mixed/F
        pool_size_fn=_pool_size,
        all_pool_sizes=all_pool_sizes,
        niveau_fn=lambda m: getattr(m, "division", "unknown"),
        team_elo_mean_fn=_team_elo_mean,
        categorie_fn=lambda _m: "Sen",  # Phase 4+ wires J02/S65 via filtre
    )


def _compute_calibration_stage(per_match: list[Any]) -> dict[str, dict[str, float]]:
    """Step 5 : per-ronde ECE aggregate (per-match avg, not Naeini-bin per-group).

    Real per-group Naeini ECE requires raw (probs, labels) per match-outcome
    not exposed by MatchStats. Aggregator (Task 10) computes the rigorous
    per-group ECE on flattened per_match across saisons.
    """
    if not per_match:
        return {}
    multicalib: dict[str, dict[str, float]] = {"by_ronde": {}}
    for r in sorted({m.ronde for m in per_match}):
        eces = [m.ece_ali for m in per_match if m.ronde == r]
        multicalib["by_ronde"][str(r)] = float(sum(eces) / len(eces)) if eces else 0.0
    return multicalib


def _compute_conformal_stage(per_match: list[Any]) -> dict[str, Any]:
    """Step 8 : split conformal CI 90% + coverage."""
    if len(per_match) < CONFORMAL_CALIB_N + 1:
        msg = f"Need >={CONFORMAL_CALIB_N + 1} matches for conformal split, got {len(per_match)}"
        raise RuntimeError(msg)
    import numpy as np

    n_calib = CONFORMAL_CALIB_N
    e_obs = np.array([m.e_score_observed for m in per_match])
    e_pred = np.array([m.e_score_predicted for m in per_match])
    cal = conformal.split_calibrate(
        e_obs[:n_calib],
        e_pred[:n_calib],
        alpha=CONFORMAL_ALPHA,
    )
    test_obs = e_obs[n_calib:]
    test_pred = e_pred[n_calib:]
    cov = conformal.coverage_rate(test_obs, test_pred, cal)
    set_size = conformal.conformal_set_size_mean(test_pred, cal)
    return {
        "coverage_global": float(cov),
        "set_size_mean": float(set_size),
        "quantile_threshold": float(cal.quantile_threshold),
        "n_calibration": cal.n_calibration,
    }


def main() -> None:
    """Per-saison D8 orchestrator entry point."""
    saison = int(os.environ["ALICE_SAISON"])
    _validate_saison(saison)
    output_dir = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _input_paths()
    logger.info("D8 saison %d — building lineage", saison)
    lineage = _build_lineage(saison, paths)

    logger.info("D8 saison %d — running BacktestRunner", saison)
    runner, backtest_report = _run_backtest(saison)
    per_match = list(backtest_report.per_match)
    logger.info("D8 saison %d — n_matches = %d", saison, len(per_match))

    if len(per_match) < CONFORMAL_CALIB_N + 1:
        msg = f"Saison {saison} only {len(per_match)} matches, need >={CONFORMAL_CALIB_N + 1}"
        raise RuntimeError(msg)

    logger.info("D8 saison %d — computing breakdowns 7 dims", saison)
    bdowns = _compute_breakdowns_stage(runner, per_match)

    logger.info("D8 saison %d — multicalibration", saison)
    multicalib = _compute_calibration_stage(per_match)

    logger.info("D8 saison %d — conformal coverage 90%%", saison)
    conformal_out = _compute_conformal_stage(per_match)

    # Stress + DRO : per-match perturbation re-runs require BacktestHarness
    # cache-mutation infrastructure (perturb opponent pool Elo / roster, re-run
    # via runner.run_single, restore cache). That infra is NOT yet built —
    # tracked as dette `D-2026-05-09-d8-perturbation-runs` in
    # `memory/project_debt_current.md`. At this stage we emit a per-saison
    # baseline summary only and flag `perturbation_closure_pending`. Aggregator
    # (Task 10) + Kaggle exec (Task 14) will surface this gap to gates G_ROB_*.
    recall_baseline_mean = float(sum(m.recall_baseline for m in per_match) / max(len(per_match), 1))
    recall_ali_mean = float(sum(m.recall_ali for m in per_match) / max(len(per_match), 1))
    logger.warning(
        "D8 saison %d — stress_elo / stress_roster / DRO perturbation closures "
        "NOT YET WIRED (dette D-2026-05-09-d8-perturbation-runs). Emitting "
        "baseline-only summary; gates G_ROB_01..05 + G_ROB_08..09 will report "
        "INCONCLUSIVE in aggregator until infra ready.",
        saison,
    )
    pending_note = "perturbation_closure_pending — cf D-2026-05-09-d8-perturbation-runs"
    stress_elo_summary = {
        "noise_levels": list(NOISE_LEVELS_ELO),
        "baseline_recall_mean": recall_baseline_mean,
        "ali_recall_mean": recall_ali_mean,
        "status": pending_note,
    }
    stress_roster_summary = {
        "turnover_rates": list(TURNOVER_RATES_ROSTER),
        "baseline_recall_mean": recall_baseline_mean,
        "ali_recall_mean": recall_ali_mean,
        "status": pending_note,
    }
    dro_summary = {
        "epsilons": list(DRO_EPSILONS),
        "n_perturbations": DRO_N_PERTURBATIONS,
        "baseline_recall_mean": recall_baseline_mean,
        "ali_recall_mean": recall_ali_mean,
        "status": pending_note,
    }

    logger.info("D8 saison %d — assembling D8SaisonReport", saison)
    report = D8SaisonReport(
        schema_version="d8.v1",
        saison=saison,
        n_matches=len(per_match),
        lineage=lineage,
        per_match=_flatten_per_match(per_match),
        breakdowns=_serialize_breakdowns(bdowns),  # type: ignore[arg-type]
        multicalibration=multicalib,
        stress_elo=stress_elo_summary,
        stress_roster=stress_roster_summary,
        conformal=conformal_out,
        dro_wasserstein=dro_summary,
    )

    output_path = output_dir / f"d8_saison_{saison}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_dump_report(report), f, indent=2, default=str, ensure_ascii=False)
    logger.info("D8 saison %d — wrote %s", saison, output_path)


def _dump_report(report: D8SaisonReport) -> dict[str, Any]:
    """JSON-safe dump of D8SaisonReport (handles nested dataclasses)."""
    out: dict[str, Any] = {}
    for f in fields(report):
        v = getattr(report, f.name)
        if is_dataclass(v):
            out[f.name] = asdict(v)
        else:
            out[f.name] = v
    return out


if __name__ == "__main__":
    main()
