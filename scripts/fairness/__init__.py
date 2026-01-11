"""Package: scripts/fairness - Fairness & Bias Detection.

Document ID: ALICE-PKG-FAIRNESS-001
Version: 2.0.0

Ce package implemente la detection et le monitoring des biais
dans les modeles ML ALICE, conformement a ISO/IEC TR 24027:2021.

Modules (refactored ISO 5055):
- types.py: BiasLevel, BiasMetrics, BiasReport
- thresholds.py: BiasThresholds, DEFAULT_THRESHOLDS
- metrics.py: compute_bias_metrics_by_group, compute_bias_by_elo_range
- checks.py: check_bias_thresholds
- report.py: generate_fairness_report
- bias_detection.py: Thin re-export (backwards compatibility)

Exports:
- BiasLevel: Enum des niveaux de biais
- BiasMetrics: Dataclass metriques par groupe
- BiasReport: Rapport complet de biais
- BiasThresholds: Seuils configurables
- DEFAULT_THRESHOLDS: Seuils par defaut
- compute_bias_metrics_by_group(): Calcul metriques
- compute_bias_by_elo_range(): Analyse par tranche Elo
- check_bias_thresholds(): Verification seuils
- generate_fairness_report(): Generation rapport

Seuils par defaut (EEOC 4/5 rule + Fairlearn):
- SPD: |SPD| < 0.1 (acceptable), >= 0.2 (critique)
- DIR: 0.8 <= DIR <= 1.25 (acceptable)
- EOD: |EOD| < 0.1 (acceptable), >= 0.2 (critique)

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- ISO/IEC 42001:2023 - AI Management System (tracabilite)
- ISO/IEC 25059:2023 - AI Quality Model
- ISO/IEC 5055:2021 - Code Quality (<300 lines per module)

See Also
--------
- docs/iso/AI_RISK_ASSESSMENT.md - Section R3: Fairness Risks
- docs/iso/STATEMENT_OF_APPLICABILITY.md - Control B.4.4
- config/hyperparameters.yaml - metrics_thresholds

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from scripts.fairness.checks import check_bias_thresholds
from scripts.fairness.metrics import compute_bias_by_elo_range, compute_bias_metrics_by_group
from scripts.fairness.report import generate_fairness_report
from scripts.fairness.thresholds import DEFAULT_THRESHOLDS, BiasThresholds
from scripts.fairness.types import BiasLevel, BiasMetrics, BiasReport

__all__ = [
    # Enums
    "BiasLevel",
    # Dataclasses
    "BiasMetrics",
    "BiasReport",
    "BiasThresholds",
    # Constants
    "DEFAULT_THRESHOLDS",
    # Functions
    "compute_bias_metrics_by_group",
    "compute_bias_by_elo_range",
    "check_bias_thresholds",
    "generate_fairness_report",
]
