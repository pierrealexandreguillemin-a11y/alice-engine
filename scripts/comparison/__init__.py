"""Module: scripts/comparison/__init__.py - Statistical Comparison Package.

Document ID: ALICE-MOD-COMPARISON-PKG-001
Version: 1.0.0

Package pour comparaison statistique de modeles ML.
Implemente McNemar 5x2cv et autres tests statistiques.

Modules:
- mcnemar_test: Test McNemar 5x2cv (Dietterich 1998)
- statistical_comparison: Pipeline de comparaison complete

ISO Compliance:
- ISO/IEC 24029:2021 - Neural Network Robustness (validation statistique)
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from scripts.comparison.mcnemar_test import (
    McNemarResult,
    mcnemar_5x2cv_test,
    mcnemar_simple_test,
)
from scripts.comparison.statistical_comparison import (
    ModelComparison,
    compare_models,
    compare_with_baseline,
)

__all__ = [
    # McNemar Tests
    "McNemarResult",
    "mcnemar_5x2cv_test",
    "mcnemar_simple_test",
    # Comparison Pipeline
    "ModelComparison",
    "compare_models",
    "compare_with_baseline",
]
