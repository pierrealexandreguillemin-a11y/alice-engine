"""Report writer + paired-statistics aggregation for the Phase 4a pilot (T9.5).

Split out of `pilot_phase4a.py` for ISO 5055 SRP (orchestrator <= 250 lines).
Holds the two pure delegated concerns:

- `aggregate_stats` : paired McNemar (recall dichotomised at the early gate) +
  Wilcoxon signed-rank (continuous recall) on the Phase 4a vs Phase 3 baseline
  rows. Defensive against the empty-input ValueError both tests raise, and
  against an all-concordant McNemar table (n_disc == 0).
- `write_report`    : emit `reports/pilot_phase4a.md` (summary + per-match table).

Document ID: ALICE-BACKTEST-PILOT-PHASE4A-HELPERS
Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.backtest.statistical import mcnemar_paired, wilcoxon_paired

# Recall threshold used to dichotomise the continuous recall into a binary
# "correct" flag for the McNemar 2x2 table (mirrors the early-gate threshold).
_DICHOTOMY = 0.50

_BASELINE_REFERENCE = 0.57  # Phase 3 N3 mean recall reference (spec §Q9 / plan).


def aggregate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired McNemar + Wilcoxon of Phase 4a recall vs the Phase 3 baseline.

    @param rows: per-match dicts with ``recall`` (Phase 4a) and
        ``recall_baseline`` (Phase 3, same match) keys.
    @returns dict with keys ``mcnemar``, ``wilcoxon`` (raw result objects or
        None) plus the flattened ``mcnemar_p``, ``mcnemar_n_disc``,
        ``wilcoxon_p``, ``wilcoxon_significant``. In the all-concordant case
        ``mcnemar_n_disc`` stays ``0`` (always an int); only ``mcnemar`` and
        ``mcnemar_p`` collapse to ``None`` (test undefined when n_disc == 0).
    """
    if not rows:
        return {
            "mcnemar": None,
            "wilcoxon": None,
            "mcnemar_p": None,
            "mcnemar_n_disc": None,
            "wilcoxon_p": None,
            "wilcoxon_significant": None,
        }

    ali_recall = [r["recall"] for r in rows]
    base_recall = [r["recall_baseline"] for r in rows]
    ali_correct = [v >= _DICHOTOMY for v in ali_recall]
    base_correct = [v >= _DICHOTOMY for v in base_recall]

    # McNemar degenerates when every pair is concordant (b + c == 0) : the
    # statsmodels table is all-zero and the test is undefined. Guard to None
    # so the pilot never crashes on a unanimous fixture (ISO 24029).
    mcnemar = None
    n_disc = sum(1 for a, b in zip(ali_correct, base_correct, strict=True) if a != b)
    if n_disc > 0:
        try:
            mcnemar = mcnemar_paired(ali_correct, base_correct)
        except (ValueError, ZeroDivisionError):
            mcnemar = None

    wilcoxon = wilcoxon_paired(ali_recall, base_recall)

    return {
        "mcnemar": mcnemar,
        "wilcoxon": wilcoxon,
        "mcnemar_p": mcnemar.p_value if mcnemar is not None else None,
        "mcnemar_n_disc": n_disc,
        "wilcoxon_p": wilcoxon.p_value,
        "wilcoxon_significant": wilcoxon.significant,
    }


def _fmt(value: Any, spec: str = ".4f") -> str:
    """Format a numeric value, falling back to 'n/a' for None."""
    if value is None:
        return "n/a"
    return format(value, spec)


def _summary_lines(summary: dict[str, Any]) -> list[str]:
    """Build the markdown summary block (no per-match table)."""
    skipped = summary["skipped"]
    return [
        "# Phase 4a Pilot — N3 saison 2024 (joint-conditional early gate)",
        "",
        "## Summary",
        "",
        f"- Viable matches run: **{summary['n_matches']}** (target {summary['max_viable']})",
        f"- Mean recall (Phase 4a): **{_fmt(summary['mean_recall'])}**",
        f"- Mean recall (Phase 3 baseline): {_fmt(summary['mean_baseline_recall'])} "
        f"(reference ~{_BASELINE_REFERENCE:.2f})",
        f"- Skipped: non_viable={skipped['non_viable']}, "
        f"no_observed={skipped['no_observed']}, "
        f"thin_residual={skipped['thin_residual']}, error={skipped['error']}",
        "",
        "> `thin_residual` = matches where draining the opponent's superior teams left too few"
        " players for 20 distinct lineups (ADR-014). A HIGH count is itself a Phase 4a finding"
        " (over-draining), not an error.",
        f"- **Wilcoxon (decisive):** p={_fmt(summary['wilcoxon_p'])}, "
        f"significant={summary['wilcoxon_significant']}",
        "",
        "> **Wilcoxon signed-rank (continuous recall) is the decisive test** (D-P3-18, SOTA)."
        " McNemar below dichotomizes recall at 0.50 and is a secondary view —"
        " a near-zero n_discordant means the two models rarely cross the 0.50 line"
        " on opposite sides, NOT an absence of effect.",
        "",
        f"- McNemar (secondary): p={_fmt(summary['mcnemar_p'])},"
        f" n_discordant={summary['mcnemar_n_disc']}"
        + (
            " (not computable — all pairs concordant)"
            if summary["mcnemar_p"] is None and summary["mcnemar_n_disc"] == 0
            else ""
        ),
        "",
        f"### Early-gate decision\n\n**{summary['decision']}**",
        "",
    ]


def _table_lines(rows: list[dict[str, Any]]) -> list[str]:
    """Build the per-match markdown table."""
    lines = [
        "## Per-match detail",
        "",
        "| opp_team | ronde | date | n_superior | recall | recall_baseline | jaccard | brier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['opp_team']} | {r['ronde']} | {r['date']} | {r['n_superior']} "
            f"| {_fmt(r['recall'])} | {_fmt(r['recall_baseline'])} "
            f"| {_fmt(r['jaccard'])} | {_fmt(r['brier'])} |"
        )
    lines.append("")
    return lines


def write_report(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Write `reports/pilot_phase4a.md` (summary block + per-match table).

    @param out_dir: directory to write into (created if absent).
    @param rows: per-match result dicts.
    @param summary: aggregated summary dict (from `run_pilot`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = _summary_lines(summary) + _table_lines(rows)
    (out_dir / "pilot_phase4a.md").write_text("\n".join(lines), encoding="utf-8")
