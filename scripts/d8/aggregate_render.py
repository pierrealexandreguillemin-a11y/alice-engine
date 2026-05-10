"""D8 aggregate rendering helpers — D8_FINDINGS.md generation.

Extracted from scripts/d8/aggregate.py (ISO 5055 ≤300 lines split).
Pure rendering : no I/O, deterministic output, easy to unit-test.

Document ID: ALICE-D8-AGGREGATE-RENDER
Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from scripts.d8.types import D8GateStatus

if TYPE_CHECKING:
    from scripts.d8.types import D8FullReport


def render_findings_md(
    report: D8FullReport,
    run_at: datetime,
    *,
    summary: dict[str, int],
) -> str:
    """Render D8_FINDINGS.md humain-readable per spec §7.4."""
    lines = [
        "# D8 Findings — Phase 3.5 STRICT\n",
        f"**Run** : {run_at.isoformat()}\n",
        f"**N matches** : {report.n_matches}\n",
        f"**Saisons** : {report.saisons}\n",
        f"**Gates G-A** : {summary['pass']}/19 PASS, "
        f"{summary['fail']} FAIL, {summary['inconclusive']} INCONCLUSIVE\n\n",
        "## Per-gate results\n\n",
    ]
    for g in report.gates_19:
        measured = "n/a" if g.status is D8GateStatus.INCONCLUSIVE else f"{g.measured_value:.4f}"
        lines.append(
            f"- **{g.gate_id}** — {g.status.value.upper()} "
            f"(measured={measured}, threshold={g.threshold:.4f}, source={g.source})\n"
        )
    lines.append("\n## Phase 4a entry gate\n\n")
    if summary["fail"] == 0 and summary["inconclusive"] == 0:
        lines.append("All 19 gates PASS — ADR-016 ready Proposed -> Accepted.\n")
    elif summary["fail"] == 0:
        lines.append(
            f"WARN {summary['inconclusive']} gates INCONCLUSIVE "
            "(missing data; perturbation infra issue or saison report partial).\n"
            "Phase 4a entry gate BLOCKED until missing values populated.\n"
        )
    else:
        lines.append(
            f"WARN {summary['fail']} gates FAIL — see D8_FAILURE_ANALYSIS_LOG.md "
            "for case-by-case decisions.\n"
        )
    return "".join(lines)
