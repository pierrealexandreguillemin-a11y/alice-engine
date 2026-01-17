#!/usr/bin/env python3
"""PostToolUse hook - Validate ISO compliance and loop corrections.

Document ID: ALICE-HOOK-POSTVALIDATE-001
Version: 1.0.0
ISO: 5055 (<50 lines/function), 42001 (tracabilite)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ISO Thresholds
AUC_MIN = 0.70
ROBUSTNESS_MIN = 0.95
FAIRNESS_MIN = 0.60  # CAUTION threshold (CRITICAL < 0.60)


def _check_training_output(cwd: Path) -> dict | None:
    """Check training output meets ISO 25059 thresholds."""
    report = cwd / "reports/autogluon_results.json"
    if not report.exists():
        return None
    data = json.loads(report.read_text())
    auc = data.get("test_auc", 0)
    if auc < AUC_MIN:
        return {
            "issue": f"AUC {auc:.4f} < {AUC_MIN} (ISO 25059)",
            "fix": "Increase time_limit or add more features",
        }
    return None


def _check_robustness_output(cwd: Path) -> dict | None:
    """Check robustness meets ISO 24029 thresholds."""
    report = cwd / "reports/iso24029_robustness.json"
    if not report.exists():
        return None
    data = json.loads(report.read_text())
    tolerance = data.get("noise_tolerance", 0)
    if tolerance < ROBUSTNESS_MIN:
        return {
            "issue": f"Robustness {tolerance:.4f} < {ROBUSTNESS_MIN} (ISO 24029)",
            "fix": "Model is fragile - retrain with data augmentation",
        }
    return None


def _check_fairness_output(cwd: Path) -> dict | None:
    """Check fairness meets ISO 24027 thresholds."""
    report = cwd / "reports/iso24027_fairness.json"
    if not report.exists():
        return None
    data = json.loads(report.read_text())
    parity = data.get("demographic_parity_ratio", 0)
    if parity < FAIRNESS_MIN:
        return {
            "issue": f"Fairness {parity:.2%} < {FAIRNESS_MIN:.0%} (ISO 24027 CRITICAL)",
            "fix": "CRITICAL bias - apply resampling or remove biased features",
        }
    return None


def main() -> None:
    """Entry point for PostToolUse hook."""
    data = json.load(sys.stdin)
    if data.get("tool_name") != "Bash":
        sys.exit(0)

    cmd = data.get("tool_input", {}).get("command", "")
    cwd = Path(data.get("cwd", "."))

    issue = None
    if "run_training" in cmd or "autogluon" in cmd.lower():
        issue = _check_training_output(cwd)
    elif "robustness" in cmd.lower():
        issue = _check_robustness_output(cwd)
    elif "fairness" in cmd.lower():
        issue = _check_fairness_output(cwd)

    if issue:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": (
                    f"ISO VIOLATION: {issue['issue']}\n"
                    f"ACTION REQUIRED: {issue['fix']}\n"
                    "Loop until resolved."
                ),
            }
        }
        print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
