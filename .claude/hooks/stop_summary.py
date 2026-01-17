#!/usr/bin/env python3
"""Stop hook - Final ISO compliance summary.

Document ID: ALICE-HOOK-STOP-001
Version: 1.0.0
ISO: 5055 (<50 lines/function), 42001 (tracabilite)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_REPORTS = [
    ("autogluon_results.json", "ISO 25059 Training"),
    ("iso24029_robustness.json", "ISO 24029 Robustness"),
    ("iso24027_fairness.json", "ISO 24027 Fairness"),
    ("iso42001_model_card.json", "ISO 42001 Model Card"),
    ("mcnemar_comparison.json", "ISO 24029 McNemar"),
]


def main() -> None:
    """Generate final ISO compliance summary."""
    data = json.load(sys.stdin)
    cwd = Path(data.get("cwd", "."))
    reports_dir = cwd / "reports"

    missing = []
    present = []
    for fname, label in REQUIRED_REPORTS:
        if (reports_dir / fname).exists():
            present.append(label)
        else:
            missing.append(label)

    # Stop hooks use top-level fields only (no hookSpecificOutput)
    if missing:
        reason = (
            f"ISO INCOMPLETE - Missing: {', '.join(missing)}"
        )
        output = {"stopReason": reason}
    else:
        # Empty object = success, no message needed
        output = {}

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
