"""D8 per-saison orchestrator — kernel entry point ALICE_SAISON.

Pipeline spec §7.2 : lineage → BacktestRunner → breakdowns 7 dims → multicalib
→ conformal 90% → real perturbation stress/DRO (perturb_runner) → JSON d8.v1.
ISO 27034/5259/5055.

Document ID: ALICE-D8-RUN
Version: 2.0.0
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import asdict, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Kaggle bootstrap : recursive glob matches depth-1 and depth-4 mount layouts.
if Path("/kaggle/input").is_dir():
    for _p in Path("/kaggle/input").glob("**/scripts/d8/run.py"):
        _root = str(_p.parents[2])
        if _root not in sys.path:
            sys.path.insert(0, _root)
            _req = Path(_root) / "scripts" / "d8" / "kaggle-requirements.txt"
            if _req.is_file():
                import subprocess  # noqa: S404

                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--quiet", "-r", str(_req)],
                    check=False,
                )  # noqa: S603
            break

from scripts.d8 import breakdowns, conformal, loader  # noqa: E402
from scripts.d8.perturb_runner import DRO_N_PERTURBATIONS  # noqa: E402
from scripts.d8.types import D8GroupBreakdown, D8Lineage, D8SaisonReport  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SAISON_MIN, SAISON_MAX = 2018, 2030
NOISE_LEVELS_ELO: tuple[float, ...] = (0.01, 0.03, 0.05, 0.07, 0.10)
TURNOVER_RATES_ROSTER: tuple[float, ...] = (0.05, 0.10, 0.20)
DRO_EPSILONS: tuple[float, ...] = (0.05, 0.10)
CONFORMAL_CALIB_N = 30
CONFORMAL_ALPHA = 0.10  # 1 - 0.90 coverage


def _validate_saison(saison: int) -> None:
    """ISO 27034 input validation on saison."""
    if not SAISON_MIN <= saison <= SAISON_MAX:
        msg = f"ALICE_SAISON {saison} outside [{SAISON_MIN}, {SAISON_MAX}]"
        raise ValueError(msg)


def _input_paths() -> dict[str, Path]:
    """Resolve input paths via env vars (Kaggle: alice-d8-input mount; local: ./)."""
    base = Path("/kaggle/input/alice-d8-input")
    mcd = Path(os.environ.get("MODEL_CACHE_DIR", base / "artefacts"))
    return {
        "joueurs": Path(os.environ.get("JOUEURS_PARQUET", base / "data/joueurs.parquet")),
        "echiquiers": Path(os.environ.get("ECHIQUIERS_PARQUET", base / "data/echiquiers.parquet")),
        "mlp": mcd / "mlp_meta_learner.joblib",
        "temp_scaler": mcd / "temperature_T.joblib",
    }


def _build_lineage(saison: int, paths: dict[str, Path]) -> D8Lineage:
    """ISO 5259 SHA-256 lineage at kernel start."""
    return D8Lineage(
        joueurs_sha256=loader.compute_file_sha256(paths["joueurs"]),
        echiquiers_sha256=loader.compute_file_sha256(paths["echiquiers"]),
        mlp_artefact_sha256=loader.compute_file_sha256(paths["mlp"]),
        temp_scaler_sha256=loader.compute_file_sha256(paths["temp_scaler"]),
        code_sha256=_resolve_code_sha(),
        ali_seed=42,
        ali_n_topk=10,
        ali_n_mc_pairs=5,
        ali_decay_lambda=0.9,
        kernel_id=f"pguillemin/d8-saison-{saison}",
        kernel_version_kaggle=os.environ.get("KAGGLE_KERNEL_VERSION", "v1"),
        run_at_utc=datetime.now(UTC).isoformat(),
    )


def _checkpoint_partial(output_dir: Path, saison: int, **stages: Any) -> None:
    """Incremental save (kaggle-deployment skill : timeout-safe artifacts)."""
    partial = {
        "schema_version": "d8.v1-partial",
        "saison": saison,
        **stages,
        "_status": "partial — perturbation stages pending",
    }
    (output_dir / f"d8_saison_{saison}_partial.json").write_text(
        json.dumps(partial, indent=2, default=str),
        encoding="utf-8",
    )


def _resolve_code_sha() -> str:
    """Read ALICE_CODE_SHA env var, fallback to CODE_SHA.txt staged by upload."""
    sha = os.environ.get("ALICE_CODE_SHA")
    if sha:
        return sha
    for p in (
        Path("/kaggle/input").glob("**/CODE_SHA.txt") if Path("/kaggle/input").is_dir() else []
    ):
        return p.read_text(encoding="utf-8").strip() or "unknown"
    return "unknown"


def _validate_per_match_finite(per_match: list[Any], saison: int) -> None:
    """ISO 27034 + ml-training-pipeline skill : reject NaN/Inf in MatchStats."""
    for m in per_match:
        for attr in ("recall_ali", "brier_ali", "ece_ali", "bss", "e_score_predicted"):
            v = getattr(m, attr, None)
            if v is None or not math.isfinite(v):
                msg = f"Saison {saison} match has non-finite {attr}={v!r}"
                raise RuntimeError(msg)


SAISON_DIVISION_FILTER: dict[int, str] = {
    # FFE renamed "Nationale III" → "Nationale 3" between saison 2021 and 2022.
    2018: "Nationale III",
    2019: "Nationale III",
    2020: "Nationale III",
    2021: "Nationale III",
    2022: "Nationale 3",
    2023: "Nationale 3",
    2024: "Nationale 3",
    2025: "Nationale 3",
    2026: "Nationale 3",
    2027: "Nationale 3",
    2028: "Nationale 3",
    2029: "Nationale 3",
    2030: "Nationale 3",
}


def _run_backtest(saison: int) -> Any:
    """Boot Plan 3 BacktestHarness + run one saison via BacktestRunner."""
    from scripts.backtest.harness import BacktestHarness
    from scripts.backtest.runner import BacktestRunner
    from scripts.backtest.runner_types import RunnerConfig

    harness = BacktestHarness()
    harness.setup()
    cfg_kwargs: dict[str, Any] = {
        "saison": saison,
        "division_filter": SAISON_DIVISION_FILTER[saison],
    }
    if (smoke := os.environ.get("SMOKE_MAX_MATCHES")) is not None:
        cfg_kwargs["max_matches"] = int(smoke)
        cfg_kwargs["stratify_min_per_ronde"] = 1  # relax for tiny smoke run
    config = RunnerConfig(**cfg_kwargs)
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
    """Step 5 : per-ronde ECE aggregate. Aggregator computes Naeini-bin global."""
    if not per_match:
        return {}
    multicalib: dict[str, dict[str, float]] = {"by_ronde": {}}
    for r in sorted({m.ronde for m in per_match}):
        eces = [m.ece_ali for m in per_match if m.ronde == r]
        multicalib["by_ronde"][str(r)] = float(sum(eces) / len(eces)) if eces else 0.0
    return multicalib


def _compute_conformal_stage(per_match: list[Any]) -> dict[str, Any]:
    """Step 8 : split conformal CI 90% + coverage."""
    import numpy as np

    e_obs = np.array([m.e_score_observed for m in per_match])
    e_pred = np.array([m.e_score_predicted for m in per_match])
    n = CONFORMAL_CALIB_N
    cal = conformal.split_calibrate(e_obs[:n], e_pred[:n], alpha=CONFORMAL_ALPHA)
    cov = conformal.coverage_rate(e_obs[n:], e_pred[n:], cal)
    set_size = conformal.conformal_set_size_mean(e_pred[n:], cal)
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
    try:
        runner, backtest_report = _run_backtest(saison)
        per_match = list(backtest_report.per_match)
    except ValueError as exc:
        logger.warning("D8 saison %d — backtest failed: %s — emitting empty report", saison, exc)
        partial = {
            "schema_version": "d8.v1-empty",
            "saison": saison,
            "n_matches": 0,
            "lineage": asdict(lineage),
            "_status": f"insufficient_matches: {exc}",
        }
        (output_dir / f"d8_saison_{saison}.json").write_text(
            json.dumps(partial, indent=2, default=str),
            encoding="utf-8",
        )
        return
    logger.info("D8 saison %d — n_matches = %d", saison, len(per_match))
    _validate_per_match_finite(per_match, saison)

    if len(per_match) < CONFORMAL_CALIB_N + 1:
        msg = f"Saison {saison} only {len(per_match)} matches, need >={CONFORMAL_CALIB_N + 1}"
        raise RuntimeError(msg)

    bdowns = _compute_breakdowns_stage(runner, per_match)
    multicalib = _compute_calibration_stage(per_match)
    conformal_out = _compute_conformal_stage(per_match)
    _checkpoint_partial(
        output_dir,
        saison,
        lineage=asdict(lineage),
        n_matches=len(per_match),
        per_match=_flatten_per_match(per_match),
        breakdowns=_serialize_breakdowns(bdowns),
        multicalibration=multicalib,
        conformal=conformal_out,
    )

    # Real perturbation closures (D-2026-05-09 RESORBE) — subset N=30 per spec §11.
    from scripts.d8.perturb_runner import (
        compute_dro_real,
        compute_stress_elo_real,
        compute_stress_roster_real,
        stratified_subset,
    )

    candidates = runner.sample_matches()
    subset = stratified_subset([c for c in candidates if c.saison == saison], seed=42)
    logger.info("D8 saison %d — stress subset N=%d", saison, len(subset))
    sb = [
        float(
            next(
                (
                    m.recall_ali
                    for m in per_match
                    if m.user_team == c.user_team and m.opponent_team == c.opp_team
                ),
                0.0,
            )
        )
        for c in subset
    ]
    stress_elo_summary = {
        "noise_levels": list(NOISE_LEVELS_ELO),
        "subset_n": len(subset),
        "outcomes": [asdict(o) for o in compute_stress_elo_real(runner, subset, sb)],
    }
    stress_roster_summary = {
        "turnover_rates": list(TURNOVER_RATES_ROSTER),
        "subset_n": len(subset),
        "outcomes": [asdict(o) for o in compute_stress_roster_real(runner, subset, sb)],
    }
    dro_summary = {
        "epsilons": list(DRO_EPSILONS),
        "n_perturbations": DRO_N_PERTURBATIONS,
        "subset_n": len(subset),
        "outcomes": {str(eps): asdict(o) for eps, o in compute_dro_real(runner, subset).items()},
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
