"""T22 driver: hold-out 2024 backtest 100 matches + JSON dump.

Plan 3 V2 T22. Produit reports/backtest/ali_holdout_2024.json avec :
- gates_summary (P3G07-P3G11)
- per_match details (lineage)
- bootstrap CIs BCa per metric
- McNemar paired ALI vs baseline Elo
- Fairness breakdown (by_ronde + by_opponent_club_size)

Robustness T14/T16 deja smoke-tested (commit 9022923, 63ba7b5).
Cite reports existants iso24029_robustness*.json.

Document ID: ALICE-BACKTEST-T22-DRIVER
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from scripts.backtest.fairness import (
    breakdown_by_opponent_club_size,
    breakdown_by_ronde,
    max_gap,
)
from scripts.backtest.harness import BacktestHarness
from scripts.backtest.runner import BacktestRunner
from scripts.backtest.runner_types import BacktestReport, RunnerConfig

logger = logging.getLogger(__name__)


def _ci_dump(ci: object) -> dict[str, object]:
    """Serialize BootstrapCI to JSON-safe dict."""
    return {
        "lower": ci.lower,  # type: ignore[attr-defined]
        "point": ci.point,  # type: ignore[attr-defined]
        "upper": ci.upper,  # type: ignore[attr-defined]
        "confidence": ci.confidence,  # type: ignore[attr-defined]
        "n_resamples": ci.n_resamples,  # type: ignore[attr-defined]
        "method": ci.method,  # type: ignore[attr-defined]
        "n_samples": ci.n_samples,  # type: ignore[attr-defined]
    }


def _opponent_pool_size(harness: BacktestHarness) -> object:
    """Return callable match -> opponent club pool size (post-hoc breakdown)."""
    cache = harness.cache
    if cache is None:
        msg = "harness.cache is None"
        raise RuntimeError(msg)
    team_to_club = cache.team_to_club
    joueurs_by_club = cache.joueurs_by_club

    def _size(m: object) -> int:
        opp_team = m.opponent_team  # type: ignore[attr-defined]
        club = team_to_club.get(opp_team)
        if club is None:
            return 0
        df = joueurs_by_club.get(club)
        return 0 if df is None else len(df)

    return _size


def _report_to_dict(report: BacktestReport) -> dict[str, object]:
    """Convert BacktestReport to JSON-safe dict."""
    return {
        "n_matches": report.n_matches,
        "gates_summary": report.gates_summary(),
        "ci_recall": _ci_dump(report.ci_recall),
        "ci_accuracy": _ci_dump(report.ci_accuracy),
        "ci_jaccard": _ci_dump(report.ci_jaccard),
        "ci_brier": _ci_dump(report.ci_brier),
        "ci_ece": _ci_dump(report.ci_ece),
        "ci_mae": _ci_dump(report.ci_mae),
        "mean_bss": report.mean_bss,
        "mcnemar": {
            "statistic": report.mcnemar.statistic,
            "p_value": report.mcnemar.p_value,
            "b_ali_only_correct": report.mcnemar.b,
            "c_baseline_only_correct": report.mcnemar.c,
            "n_discordant": report.mcnemar.n_discordant,
            "method": report.mcnemar.method,
            "significant": report.mcnemar.significant,
        },
        "per_match": [asdict(m) for m in report.per_match],
    }


def main() -> None:
    """Run T22 hold-out 2024 backtest + dump JSON."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    out_dir = Path("reports/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ali_holdout_2024.json"

    logger.info("T22: setup harness")
    harness = BacktestHarness()
    harness.setup()

    config = RunnerConfig(
        saison=2024,
        rondes=(1, 3, 5, 7, 9, 11),
        max_matches=100,
        team_size=8,
        division="N3",
        nb_rondes_total=11,
        seed=42,
        n_bootstrap=1000,
        bootstrap_confidence=0.95,
        skip_failed_matches=True,
    )

    logger.info(
        "T22: run %d matches saison %d rondes %s", config.max_matches, config.saison, config.rondes
    )
    runner = BacktestRunner(harness=harness, config=config)
    report = runner.run()
    logger.info("T22: %d matches succeeded", report.n_matches)

    logger.info("T22: fairness breakdown")
    by_ronde = breakdown_by_ronde(report.per_match)
    by_size = breakdown_by_opponent_club_size(report.per_match, _opponent_pool_size(harness))

    payload = {
        "config": asdict(config),
        "report": _report_to_dict(report),
        "fairness": {
            "by_ronde": {k: asdict(v) for k, v in by_ronde.items()},
            "by_opponent_club_size": {k: asdict(v) for k, v in by_size.items()},
            "max_gap_recall_by_ronde": max_gap(by_ronde),
            "max_gap_recall_by_size": max_gap(by_size),
        },
    }

    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("T22: wrote %s", out_path)

    summary = report.gates_summary()
    print("\n=== T22 GATES SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\nN matches: {report.n_matches}")
    print(
        f"Recall  CI BCa: [{report.ci_recall.lower:.4f}, {report.ci_recall.upper:.4f}] point {report.ci_recall.point:.4f}"
    )
    print(
        f"Jaccard CI BCa: [{report.ci_jaccard.lower:.4f}, {report.ci_jaccard.upper:.4f}] point {report.ci_jaccard.point:.4f}"
    )
    print(
        f"Brier   CI BCa: [{report.ci_brier.lower:.4f}, {report.ci_brier.upper:.4f}] point {report.ci_brier.point:.4f}"
    )
    print(
        f"ECE     CI BCa: [{report.ci_ece.lower:.4f}, {report.ci_ece.upper:.4f}] point {report.ci_ece.point:.4f}"
    )
    print(
        f"MAE     CI BCa: [{report.ci_mae.lower:.4f}, {report.ci_mae.upper:.4f}] point {report.ci_mae.point:.4f}"
    )
    print(f"Mean BSS: {report.mean_bss:.4f}")
    print(
        f"McNemar p={report.mcnemar.p_value:.4g}  b(ALI+only)={report.mcnemar.b} c(base+only)={report.mcnemar.c} method={report.mcnemar.method}"
    )


if __name__ == "__main__":
    main()
