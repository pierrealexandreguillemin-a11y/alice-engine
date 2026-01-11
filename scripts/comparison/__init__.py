"""Package: scripts/comparison - Statistical Comparison.

Document ID: ALICE-MOD-COMPARISON-PKG-001
Version: 2.0.0

Package pour comparaison statistique de modeles ML.
Implemente McNemar 5x2cv et autres tests statistiques.

Modules (refactored ISO 5055):
- mcnemar_test: Test McNemar 5x2cv (Dietterich 1998)
- types.py: ModelComparison dataclass
- metrics.py: compute_metrics
- recommendation.py: generate_recommendation
- core.py: compare_models
- baseline.py: compare_with_baseline
- report.py: save_comparison_report
- pipeline.py: full_comparison_pipeline
- statistical_comparison.py: Thin re-export (backwards compatibility)

ISO Compliance:
- ISO/IEC 24029:2021 - Neural Network Robustness (validation statistique)
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes par module)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from scripts.comparison.baseline import compare_with_baseline
from scripts.comparison.core import compare_models
from scripts.comparison.mcnemar_test import (
    McNemarResult,
    mcnemar_5x2cv_test,
    mcnemar_simple_test,
)
from scripts.comparison.pipeline import full_comparison_pipeline
from scripts.comparison.report import save_comparison_report
from scripts.comparison.types import ModelComparison

__all__ = [
    # McNemar Tests
    "McNemarResult",
    "mcnemar_5x2cv_test",
    "mcnemar_simple_test",
    # Types
    "ModelComparison",
    # Comparison Functions
    "compare_models",
    "compare_with_baseline",
    "full_comparison_pipeline",
    "save_comparison_report",
]
