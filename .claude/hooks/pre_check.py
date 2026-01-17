#!/usr/bin/env python3
"""PreToolUse hook - Block execution if conditions not met.

Document ID: ALICE-HOOK-PRECHECK-001
Version: 1.2.0
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


def _check_data(cwd: Path) -> list[str]:
    """Check ISO 5259 data files exist."""
    return [f"MISSING: {cwd / 'data/features' / f'{n}.parquet'}" for n in ("train", "valid", "test") if not (cwd / "data/features" / f"{n}.parquet").exists()]


def _check_ram(min_gb: float = 6.0) -> list[str]:
    """Check available RAM meets minimum requirement."""
    if _psutil is None:
        return []
    avail = _psutil.virtual_memory().available / 1e9
    return [f"RAM: {avail:.1f}GB < {min_gb}GB"] if avail < min_gb else []


def _check_deps() -> list[str]:
    """Check AutoGluon dependencies are installed."""
    missing = [n for n in ("fastai", "autogluon.tabular") if importlib.util.find_spec(n) is None]
    return [f"MISSING: {n}" for n in missing]


def _check_model(cwd: Path, subpath: str) -> list[str]:
    """Check model directory exists."""
    path = cwd / "models" / subpath
    return [f"MISSING: {path}"] if not path.exists() else []


def _check_baseline(cwd: Path) -> list[str]:
    """Check baseline models exist."""
    return ["MISSING: baseline models"] if not list((cwd / "models").glob("v*/metadata.json")) else []


def _check_reports(cwd: Path, names: list[str]) -> list[str]:
    """Check required report files exist."""
    return [f"MISSING: {n}" for n in names if not (cwd / "reports" / n).exists()]


def main() -> None:
    """Entry point for PreToolUse hook."""
    data = json.load(sys.stdin)
    if data.get("tool_name") != "Bash":
        sys.exit(0)

    cmd, cwd = data.get("tool_input", {}).get("command", ""), Path(data.get("cwd", "."))
    errors: list[str] = []

    if "train_models_parallel" in cmd:
        errors = _check_data(cwd) + _check_ram()
    elif "TabularPredictor" in cmd and "fit" in cmd:
        errors = _check_deps() + _check_data(cwd) + _check_ram()
    elif "validate_robustness" in cmd or "iso_robustness" in cmd:
        errors = _check_model(cwd, "autogluon/autogluon_extreme_v2") + _check_data(cwd)
    elif "validate_fairness" in cmd or "iso_fairness" in cmd:
        errors = _check_model(cwd, "autogluon/autogluon_extreme_v2")
    elif "mcnemar" in cmd.lower():
        errors = _check_model(cwd, "autogluon/autogluon_extreme_v2") + _check_baseline(cwd)
    elif "ISO_25059" in cmd:
        errors = _check_reports(cwd, ["autogluon_results.json", "robustness_report.json", "fairness_report.json", "mcnemar_comparison.json"])

    if errors:
        for err in errors:
            print(f"BLOCK: {err}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
