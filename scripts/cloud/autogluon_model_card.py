"""AutoGluon Model Card builder — ISO 42001 (ISO 5055 SRP split)."""

from __future__ import annotations

import os
import platform
import sys
from datetime import UTC, datetime
from typing import Any


def build_model_card(
    metrics: dict,
    lineage: dict,
    robustness: dict,
    fairness: dict,
    leaderboard_len: int,
    best_model: str,
    version: str,
    config: dict[str, Any],
    auc_floor: float,
) -> dict:
    """ISO 42001 Model Card with full traceability."""
    pkgs: dict[str, str] = {}
    for pkg in ("autogluon", "pandas", "scikit-learn", "numpy"):
        try:
            import importlib.metadata as im  # noqa: PLC0415

            pkgs[pkg] = im.version(pkg)
        except Exception:
            pkgs[pkg] = "unknown"
    return {
        "version": version,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "status": "CANDIDATE",
        "model_type": "AutoGluon_ensemble",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "kaggle_kernel": os.environ.get("KAGGLE_KERNEL_RUN_SLUG", "local"),
            "packages": pkgs,
        },
        "config": config,
        "data_lineage": lineage,
        "metrics": metrics,
        "best_model": best_model,
        "num_models_trained": leaderboard_len,
        "iso_24029_robustness": robustness,
        "iso_24027_fairness": fairness,
        "quality_gate": {"passed": metrics["test_auc"] >= auc_floor, "auc_floor": auc_floor},
        "limitations": ["FFE interclub data only", "Not for tournament games"],
        "conformance": {
            "ISO_42001": "CANDIDATE",
            "ISO_5259": "COMPLIANT",
            "ISO_24029": robustness["status"],
            "ISO_24027": fairness["status"],
        },
    }
