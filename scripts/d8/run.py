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


def _resolve_input_base() -> Path:
    """Probe Kaggle mount layout (depth-1 OR depth-4 for freshly-uploaded datasets).

    kaggle-deployment skill 2026-05-04 : fresh dataset_sources mount at depth-4
    `/kaggle/input/datasets/{user}/{slug}/`, NOT depth-1 `/kaggle/input/{slug}/`.
    Probe both, fail-fast if neither contains joueurs.parquet.
    """
    candidates = [
        Path("/kaggle/input/alice-d8-input"),
        Path("/kaggle/input/datasets/pguillemin/alice-d8-input"),
    ]
    # Robust glob fallback in case Kaggle layout shifts again
    for p in (
        Path("/kaggle/input").glob("**/data/joueurs.parquet")
        if Path("/kaggle/input").is_dir()
        else []
    ):
        candidates.append(p.parents[1])
    for base in candidates:
        if (base / "data" / "joueurs.parquet").is_file():
            return base
    return Path("/kaggle/input/alice-d8-input")  # local dev fallback


def _input_paths() -> dict[str, Path]:
    """Resolve input paths via env vars (Kaggle: alice-d8-input mount; local: ./)."""
    base = _resolve_input_base()
    mcd = Path(os.environ.get("MODEL_CACHE_DIR", base / "artefacts"))
    return {
        "joueurs": Path(os.environ.get("JOUEURS_PARQUET", base / "data/joueurs.parquet")),
        "echiquiers": Path(os.environ.get("ECHIQUIERS_PARQUET", base / "data/echiquiers.parquet")),
        "mlp": mcd / "mlp_meta_learner.joblib",
        "temp_scaler": mcd / "temperature_T.joblib",
    }


def _build_lineage(saison: int, paths: dict[str, Path]) -> D8Lineage:
    """ISO 5259 SHA-256 lineage at kernel start."""
    division = os.environ.get("ALICE_DIVISION", SAISON_DIVISION_FILTER[saison])
    div_slug = division.lower().replace(" ", "-")
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
        kernel_id=f"pguillemin/d8-{saison}-{div_slug}",
        kernel_version_kaggle=os.environ.get("KAGGLE_KERNEL_VERSION", "v1"),
        run_at_utc=datetime.now(UTC).isoformat(),
        division=division,
        saison=saison,
    )


def _checkpoint_partial(output_dir: Path, saison: int, div_slug: str, **stages: Any) -> None:
    """Incremental save (kaggle-deployment skill : timeout-safe artifacts)."""
    partial = {
        "schema_version": "d8.v1-partial",
        "saison": saison,
        "division_slug": div_slug,
        **stages,
        "_status": "partial — perturbation stages pending",
    }
    (output_dir / f"d8_{saison}_{div_slug}_partial.json").write_text(
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

# type_competition naming change FFE (parser pre-2022 tagged "autre", post tag "national").
SAISON_TYPE_COMPETITION: dict[int, str] = {
    2018: "autre",
    2019: "autre",
    2020: "autre",
    2021: "autre",
    2022: "national",
    2023: "national",
    2024: "national",
    2025: "national",
    2026: "national",
    2027: "national",
    2028: "national",
    2029: "national",
    2030: "national",
}

# ADR-021 : Division-specific rondes filter.
# Nationale 1-4 format championnat : rondes 5,7,9,11 (fin-saison reporting).
# Top 16 format élite : 7 rondes régulière (Groupe A/B) + 4 rondes finale
# (Poule Haute/Basse). All rondes 1-7 must be included → 88 candidates,
# satisfait CONFORMAL_CALIB_N=30 invariant (line 350).
# Without override, rondes_default=(5,7,9,11) filtre 88→16 matches Top 16
# → RuntimeError ligne 368 uncaught → kernel ERROR (Phase A v3 Top 16 2026-05-12).
DIVISION_RONDES_DEFAULT: dict[str, tuple[int, ...]] = {
    "Top 16": (1, 2, 3, 4, 5, 6, 7),
}


CURRENT_SAISON = 2024  # ALIDataCache from_parquets() reflects this saison's roster


def _run_backtest(saison: int) -> Any:
    """Boot Plan 3 BacktestHarness + run one (saison, division) via BacktestRunner.

    Reads `ALICE_DIVISION` env var to override default mapping.
    Routes to historical cache reconstruction (ADR-018) when saison != CURRENT_SAISON.
    """
    from scripts.backtest.harness import BacktestHarness
    from scripts.backtest.runner import BacktestRunner
    from scripts.backtest.runner_types import RunnerConfig

    harness = BacktestHarness()
    harness.setup(historical_saison=None if saison == CURRENT_SAISON else saison)

    division = os.environ.get("ALICE_DIVISION", SAISON_DIVISION_FILTER[saison])
    type_comp = SAISON_TYPE_COMPETITION[saison]
    # ADR-021 : division-specific override (Top 16 = 7 rondes), default fallback
    # Nationale 1-4 = (5,7,9,11) post-2022 / (1,3,5,7,9) pre-2022.
    if division in DIVISION_RONDES_DEFAULT:
        rondes_default = DIVISION_RONDES_DEFAULT[division]
    else:
        rondes_default = (5, 7, 9, 11) if saison >= 2022 else (1, 3, 5, 7, 9)  # noqa: PLR2004

    cfg_kwargs: dict[str, Any] = {
        "saison": saison,
        "division_filter": division,
        "type_competition": type_comp,
        "rondes": rondes_default,
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


def _compute_conformal_stage(per_match: list[Any], support_max: float = 8.0) -> dict[str, Any]:
    """Step 8 : split conformal CI 90% + coverage.

    @param support_max: upper bound of e_score support (D-2026-05-11). For
        Phase A all 5 divisions team_size=8 → support_max=8.0.
    """
    import numpy as np

    e_obs = np.array([m.e_score_observed for m in per_match])
    e_pred = np.array([m.e_score_predicted for m in per_match])
    n = CONFORMAL_CALIB_N
    cal = conformal.split_calibrate(e_obs[:n], e_pred[:n], alpha=CONFORMAL_ALPHA)
    cov = conformal.coverage_rate(e_obs[n:], e_pred[n:], cal)
    set_size = conformal.conformal_set_size_mean(e_pred[n:], cal, support_max=support_max)
    return {
        "coverage_global": float(cov),
        "set_size_mean": float(set_size),
        "set_size_relative": float(set_size / support_max),
        "support_max": float(support_max),
        "quantile_threshold": float(cal.quantile_threshold),
        "n_calibration": cal.n_calibration,
    }


def main() -> None:
    """Per-(saison, division) D8 orchestrator entry point."""
    saison = int(os.environ["ALICE_SAISON"])
    _validate_saison(saison)
    division = os.environ.get("ALICE_DIVISION", SAISON_DIVISION_FILTER[saison])
    div_slug = division.lower().replace(" ", "-")
    output_dir = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _input_paths()
    logger.info("D8 saison %d %s — building lineage", saison, division)
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
        (output_dir / f"d8_{saison}_{div_slug}.json").write_text(
            json.dumps(partial, indent=2, default=str),
            encoding="utf-8",
        )
        return
    logger.info("D8 saison %d — n_matches = %d", saison, len(per_match))
    _validate_per_match_finite(per_match, saison)

    if len(per_match) < CONFORMAL_CALIB_N + 1:
        msg = f"Saison {saison} only {len(per_match)} matches, need >={CONFORMAL_CALIB_N + 1}"
        if os.environ.get("SMOKE_MAX_MATCHES") is not None:
            # Smoke mode: emit partial report (skip downstream stages requiring N>=31)
            partial = {
                "schema_version": "d8.v1-smoke",
                "saison": saison,
                "n_matches": len(per_match),
                "lineage": asdict(lineage),
                "per_match": _flatten_per_match(per_match),
                "_status": f"smoke_below_conformal_threshold: {msg}",
            }
            (output_dir / f"d8_{saison}_{div_slug}.json").write_text(
                json.dumps(partial, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("D8 saison %d smoke partial written (n=%d)", saison, len(per_match))
            return
        raise RuntimeError(msg)

    bdowns = _compute_breakdowns_stage(runner, per_match)
    multicalib = _compute_calibration_stage(per_match)
    conformal_out = _compute_conformal_stage(per_match, support_max=float(runner.config.team_size))
    _checkpoint_partial(
        output_dir,
        saison,
        div_slug,
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

    output_path = output_dir / f"d8_{saison}_{div_slug}.json"
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
