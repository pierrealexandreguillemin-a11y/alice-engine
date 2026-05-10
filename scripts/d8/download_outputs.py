"""Download d8-aggregator outputs to reports/d8/.

Pulls 4 artefacts from the d8-aggregator Kaggle kernel output bucket :
- d8_full_report.json (schema d8-aggregator.v1)
- D8_FINDINGS.md (humain)
- D8_FAILURE_ANALYSIS_LOG.md (case-by-case template)
- gates_19_status.json

Document ID: ALICE-D8-DOWNLOAD
Version: 1.0.0
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
KERNEL_SLUG = "pguillemin/d8-aggregator"
EXPECTED_OUTPUTS: tuple[str, ...] = (
    "d8_full_report.json",
    "D8_FINDINGS.md",
    "D8_FAILURE_ANALYSIS_LOG.md",
    "gates_19_status.json",
)


def main() -> int:
    """Download aggregator outputs + validate completeness."""
    out_dir = REPO / "reports" / "d8"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 - args are static literals
        ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(out_dir)],
        check=True,
    )
    missing = [f for f in EXPECTED_OUTPUTS if not (out_dir / f).exists()]
    if missing:
        msg = f"Missing aggregator outputs in {out_dir}: {missing}"
        raise FileNotFoundError(msg)
    sys.stdout.write(f"D8 outputs downloaded to {out_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
