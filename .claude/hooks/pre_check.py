#!/usr/bin/env python3
"""PreToolUse hook - Block if prerequisites missing with correction guidance.

Document ID: ALICE-HOOK-PRECHECK-001
Version: 2.0.0
ISO: 5055 (<50 lines/function), 27034 (secure coding)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psutil as psutil_type

_psutil: psutil_type | None = None
if importlib.util.find_spec("psutil"):
    import psutil as _psutil  # type: ignore[no-redef]


def _check_data(cwd: Path) -> list[tuple[str, str]]:
    """Check ISO 5259 data files exist."""
    errors = []
    for name in ("train", "valid", "test"):
        path = cwd / "data/features" / f"{name}.parquet"
        if not path.exists():
            errors.append((f"MISSING: {path}", "Run feature engineering first"))
    return errors


def _check_ram(min_gb: float = 6.0) -> list[tuple[str, str]]:
    """Check available RAM meets minimum requirement."""
    if _psutil is None:
        return []
    avail = _psutil.virtual_memory().available / 1e9
    if avail < min_gb:
        return [(f"RAM: {avail:.1f}GB < {min_gb}GB", "Close applications to free RAM")]
    return []


def _check_autogluon_model(cwd: Path) -> list[tuple[str, str]]:
    """Check AutoGluon model exists."""
    ag_dir = cwd / "models/autogluon"
    if not ag_dir.exists() or not list(ag_dir.glob("*/predictor.pkl")):
        return [("MISSING: AutoGluon model", "Run: python -m scripts.autogluon.run_training")]
    return []


def _check_baseline(cwd: Path) -> list[tuple[str, str]]:
    """Check baseline models exist."""
    if not list((cwd / "models").glob("v*/metadata.json")):
        return [("MISSING: baseline models", "Run: python -m scripts.train_models_parallel")]
    return []


def _check_reports(cwd: Path, names: list[str]) -> list[tuple[str, str]]:
    """Check required report files exist."""
    errors = []
    for name in names:
        if not (cwd / "reports" / name).exists():
            errors.append((f"MISSING: reports/{name}", f"Generate {name} first"))
    return errors


def main() -> None:
    """Entry point for PreToolUse hook."""
    data = json.load(sys.stdin)
    if data.get("tool_name") != "Bash":
        sys.exit(0)

    cmd = data.get("tool_input", {}).get("command", "")
    cwd = Path(data.get("cwd", "."))
    errors: list[tuple[str, str]] = []

    # Training commands
    if "run_training" in cmd or "train_models_parallel" in cmd:
        errors = _check_data(cwd) + _check_ram()
    elif "TabularPredictor" in cmd and "fit" in cmd:
        errors = _check_data(cwd) + _check_ram()
    # Validation commands
    elif "robustness" in cmd.lower() or "iso_robustness" in cmd:
        errors = _check_autogluon_model(cwd) + _check_data(cwd)
    elif "fairness" in cmd.lower() or "iso_fairness" in cmd:
        errors = _check_autogluon_model(cwd)
    # Comparison commands
    elif "mcnemar" in cmd.lower():
        errors = _check_autogluon_model(cwd) + _check_baseline(cwd)
    # Report generation
    elif "iso25059" in cmd.lower() or "generate_iso" in cmd.lower():
        errors = _check_reports(cwd, [
            "autogluon_results.json",
            "iso24029_robustness.json",
            "iso24027_fairness.json",
            "mcnemar_comparison.json",
        ])

    if errors:
        issues = [e[0] for e in errors]
        fixes = [e[1] for e in errors]
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "\n".join(issues),
                "additionalContext": (
                    "BLOCKED - Prerequisites missing:\n"
                    + "\n".join(f"  - {i}" for i in issues)
                    + "\n\nFIX:\n"
                    + "\n".join(f"  - {f}" for f in fixes)
                ),
            }
        }
        print(json.dumps(output))
        sys.exit(2)


if __name__ == "__main__":
    main()
