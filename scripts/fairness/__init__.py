"""Package: scripts/fairness - Fairness & Bias Detection.

Document ID: ALICE-PKG-FAIRNESS-001
Version: 1.0.0

Ce package implémente la détection et le monitoring des biais
dans les modèles ML ALICE, conformément à ISO/IEC TR 24027:2021.

Modules:
- bias_detection.py: Métriques de fairness par groupe démographique
  - SPD (Statistical Parity Difference)
  - EOD (Equal Opportunity Difference)
  - DIR (Disparate Impact Ratio)

Exports:
- BiasMetrics: Dataclass métriques par groupe
- BiasReport: Rapport complet de biais
- BiasThresholds: Seuils configurables
- compute_bias_metrics_by_group(): Calcul métriques
- compute_bias_by_elo_range(): Analyse par tranche Elo
- check_bias_thresholds(): Vérification seuils
- generate_fairness_report(): Génération rapport

Seuils par défaut (EEOC 4/5 rule + Fairlearn):
- SPD: |SPD| < 0.1 (acceptable), >= 0.2 (critique)
- DIR: 0.8 <= DIR <= 1.25 (acceptable)
- EOD: |EOD| < 0.1 (acceptable), >= 0.2 (critique)

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- ISO/IEC 42001:2023 - AI Management System (traçabilité)
- ISO/IEC 25059:2023 - AI Quality Model
- ISO/IEC 5055:2021 - Code Quality

See Also
--------
- docs/iso/AI_RISK_ASSESSMENT.md - Section R3: Fairness Risks
- docs/iso/STATEMENT_OF_APPLICABILITY.md - Control B.4.4
- config/hyperparameters.yaml - metrics_thresholds

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from scripts.fairness.bias_detection import (
    BiasLevel,
    BiasMetrics,
    BiasReport,
    BiasThresholds,
    check_bias_thresholds,
    compute_bias_by_elo_range,
    compute_bias_metrics_by_group,
    generate_fairness_report,
)

__all__ = [
    # Enums
    "BiasLevel",
    # Dataclasses
    "BiasMetrics",
    "BiasReport",
    "BiasThresholds",
    # Functions
    "compute_bias_metrics_by_group",
    "compute_bias_by_elo_range",
    "check_bias_thresholds",
    "generate_fairness_report",
]
