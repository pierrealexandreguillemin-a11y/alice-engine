"""Module: scripts/comparison/statistical_comparison.py - Thin re-export.

Document ID: ALICE-MOD-STATCOMP-001
Version: 2.0.0

DEPRECATED: Ce module est conserve pour compatibilite ascendante.
Importer directement depuis scripts.comparison.

Le module original a ete refactore en modules SRP (ISO 5055):
- types.py: ModelComparison dataclass
- metrics.py: compute_metrics
- recommendation.py: generate_recommendation
- core.py: compare_models
- baseline.py: compare_with_baseline
- report.py: save_comparison_report
- pipeline.py: full_comparison_pipeline

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (SRP, <50 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from scripts.comparison.baseline import compare_with_baseline
from scripts.comparison.core import compare_models
from scripts.comparison.pipeline import full_comparison_pipeline
from scripts.comparison.recommendation import (
    _generate_recommendation,
    generate_recommendation,
)
from scripts.comparison.report import save_comparison_report
from scripts.comparison.types import ModelComparison

__all__ = [
    "ModelComparison",
    "compare_models",
    "compare_with_baseline",
    "full_comparison_pipeline",
    "save_comparison_report",
    "generate_recommendation",
    "_generate_recommendation",
]
