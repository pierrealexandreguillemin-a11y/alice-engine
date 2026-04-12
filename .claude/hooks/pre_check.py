#!/usr/bin/env python3
"""PreToolUse hook - Block if prerequisites missing with correction guidance.

Document ID: ALICE-HOOK-PRECHECK-001
Version: 3.0.0
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


def _check_kaggle_push(cwd: Path, cmd: str) -> list[tuple[str, str]]:
    """Block kaggle push without pre-flight verification."""
    errors = []

    # Check kernel-metadata.json exists and is readable
    meta_path = cwd / "scripts/cloud/kernel-metadata.json"
    if not meta_path.exists():
        errors.append(("MISSING: kernel-metadata.json", "Copy from kernel-metadata-*.json"))
        return errors

    meta = json.loads(meta_path.read_text())
    slug = meta.get("id", "")

    # Dataset sources must exist on Kaggle (can't verify remotely, but check local upload script ran)
    ds_sources = meta.get("dataset_sources", [])
    if ds_sources:
        code_ds = [d for d in ds_sources if "code" in d]
        if code_ds:
            # Check upload_all_data.py was run (cloud/__init__.py is created by it)
            cloud_init = cwd / "scripts/cloud/__init__.py"
            if not cloud_init.exists():
                errors.append(
                    (
                        "Dataset alice-code may not be uploaded",
                        "Run: python -m scripts.cloud.upload_all_data",
                    )
                )

    # Check enable_internet only for kernels that need pip install at runtime
    # V9 training kernels use only pre-installed packages (no pip install needed)
    needs_internet = "autogluon" in slug or "sft" in slug or "unsloth" in slug
    if needs_internet and not meta.get("enable_internet"):
        errors.append(("enable_internet=false", "Training kernels need pip install at runtime"))

    # Warn about API limitations
    if meta.get("accelerator"):
        errors.append(
            (
                f"accelerator={meta['accelerator']} in metadata is IGNORED by Kaggle API",
                "Set accelerator from Kaggle UI: Edit → Session options",
            )
        )

    # Check script file exists
    code_file = meta.get("code_file", "")
    if code_file and not (cwd / "scripts/cloud" / code_file).exists():
        errors.append((f"MISSING: scripts/cloud/{code_file}", "Check code_file in metadata"))

    # Check slug looks intentional (has alice- prefix)
    if slug and not slug.startswith("pguillemin/alice-"):
        errors.append(
            (
                "Kernel slug not standard",
                "Slug should start with pguillemin/alice-",
            )
        )

    return errors


def _check_kaggle_dataset_upload(cwd: Path) -> list[tuple[str, str]]:
    """Verify data files exist before uploading to Kaggle."""
    errors = []
    for name in ("echiquiers.parquet", "joueurs.parquet"):
        if not (cwd / "data" / name).exists():
            errors.append((f"MISSING: data/{name}", "Run: make refresh-data"))
    return errors


_ISO_REPORT_FILES = [
    "autogluon_results.json",
    "iso24029_robustness.json",
    "iso24027_fairness.json",
    "mcnemar_comparison.json",
]

# (keyword_in_cmd, keyword_in_lower, checker_function)
_ROUTES: list[tuple[list[str], list[str], str]] = [
    (["kernels_push", "kernels.push", "kernels push"], [], "kaggle_push"),
    (["upload_all_data"], [], "kaggle_upload"),
    (["run_training", "train_models_parallel"], [], "training"),
    (["TabularPredictor"], ["fit"], "training"),
    ([], ["robustness", "iso_robustness"], "robustness"),
    ([], ["fairness", "iso_fairness"], "fairness"),
    ([], ["mcnemar"], "comparison"),
    ([], ["iso25059", "generate_iso"], "reports"),
]


def _route_checks(cmd: str, cwd: Path) -> list[tuple[str, str]]:  # noqa: PLR0911
    """Route command to appropriate checks."""
    cl = cmd.lower()
    for exact_keys, lower_keys, check_type in _ROUTES:
        exact_match = any(k in cmd for k in exact_keys) if exact_keys else True
        lower_match = any(k in cl for k in lower_keys) if lower_keys else True
        if not (exact_match and lower_match):
            continue
        if check_type == "kaggle_push":
            return _check_kaggle_push(cwd, cmd)
        if check_type == "kaggle_upload":
            return _check_kaggle_dataset_upload(cwd)
        if check_type == "training":
            return _check_data(cwd) + _check_ram()
        if check_type == "robustness":
            return _check_autogluon_model(cwd) + _check_data(cwd)
        if check_type == "fairness":
            return _check_autogluon_model(cwd)
        if check_type == "comparison":
            return _check_autogluon_model(cwd) + _check_baseline(cwd)
        if check_type == "reports":
            return _check_reports(cwd, _ISO_REPORT_FILES)
    return []


def _deny(errors: list[tuple[str, str]]) -> None:
    """Emit deny decision and exit."""
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


def main() -> None:
    """Entry point for PreToolUse hook."""
    data = json.load(sys.stdin)
    if data.get("tool_name") != "Bash":
        sys.exit(0)
    cmd = data.get("tool_input", {}).get("command", "")
    cwd = Path(data.get("cwd", "."))
    errors = _route_checks(cmd, cwd)
    if errors:
        _deny(errors)


if __name__ == "__main__":
    main()
