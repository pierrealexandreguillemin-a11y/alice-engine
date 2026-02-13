"""ISO knowledge base data for ALICE semantic memory.

Contains the ISO standards definitions, thresholds, and mitigation
strategies used by ISOSemanticMemory.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management
- ISO/IEC TR 24027:2021 - Bias in AI
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC 42005:2025 - AI Impact Assessment

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

from scripts.agents.semantic_memory import ISOStandard, ISOThreshold, MitigationStrategy


def load_iso_knowledge() -> dict[str, ISOStandard]:
    """Load ISO knowledge base with standards, thresholds and mitigations."""
    return {
        "fairness": _build_fairness_standard(),
        "robustness": _build_robustness_standard(),
        "impact": _build_impact_standard(),
    }


def _build_fairness_standard() -> ISOStandard:
    """Build ISO/IEC TR 24027:2021 - Bias in AI standard."""
    return ISOStandard(
        code="ISO/IEC TR 24027:2021",
        name="Bias in AI",
        version="2021",
        thresholds=[
            ISOThreshold(
                metric="demographic_parity_ratio", compliant=0.80, caution=0.60, critical=0.50
            ),
            ISOThreshold(
                metric="equalized_odds_ratio", compliant=0.80, caution=0.60, critical=0.50
            ),
        ],
        mitigations=_fairness_mitigations(),
        reference_url="https://www.iso.org/standard/77607.html",
    )


def _fairness_mitigations() -> list[MitigationStrategy]:
    """Return fairness mitigation strategies (ISO TR 24027)."""
    return [
        MitigationStrategy(
            name="Reweighting",
            phase="pre-processing",
            description="Ponderer les echantillons pour equilibrer les groupes",
            effectiveness="high",
            implementation="from fairlearn.preprocessing import Reweighing",
        ),
        MitigationStrategy(
            name="Resampling",
            phase="pre-processing",
            description="Sur/sous-echantillonner les groupes desavantages",
            effectiveness="medium",
            implementation="from imblearn.over_sampling import SMOTE",
        ),
        MitigationStrategy(
            name="Fairness Constraint",
            phase="in-processing",
            description="Ajouter contrainte d'equite a la fonction de perte",
            effectiveness="high",
            implementation="ThresholdOptimizer from fairlearn",
        ),
        MitigationStrategy(
            name="Threshold Adjustment",
            phase="post-processing",
            description="Ajuster les seuils de decision par groupe",
            effectiveness="medium",
            implementation="CalibratedEqualizedOdds from fairlearn",
        ),
    ]


def _build_robustness_standard() -> ISOStandard:
    """Build ISO/IEC 24029:2021 - Neural Network Robustness standard."""
    return ISOStandard(
        code="ISO/IEC 24029:2021",
        name="Neural Network Robustness",
        version="2021/2023",
        thresholds=[
            ISOThreshold(metric="noise_tolerance", compliant=0.95, caution=0.90, critical=0.85),
            ISOThreshold(metric="consistency_rate", compliant=0.95, caution=0.90, critical=0.85),
            ISOThreshold(
                metric="feature_dropout_resilience", compliant=0.90, caution=0.80, critical=0.70
            ),
        ],
        mitigations=_robustness_mitigations(),
        reference_url="https://www.iso.org/standard/77609.html",
    )


def _robustness_mitigations() -> list[MitigationStrategy]:
    """Return robustness mitigation strategies (ISO 24029)."""
    return [
        MitigationStrategy(
            name="Data Augmentation",
            phase="pre-processing",
            description="Augmenter les donnees avec bruit controle",
            effectiveness="high",
            implementation="Add Gaussian noise to numeric features",
        ),
        MitigationStrategy(
            name="Ensemble Methods",
            phase="in-processing",
            description="Utiliser des ensembles pour robustesse",
            effectiveness="high",
            implementation="AutoGluon WeightedEnsemble",
        ),
        MitigationStrategy(
            name="Adversarial Training",
            phase="in-processing",
            description="Entrainer sur exemples adversariaux",
            effectiveness="medium",
            implementation="adversarial_training module",
        ),
    ]


def _build_impact_standard() -> ISOStandard:
    """Build ISO/IEC 42005:2025 - AI Impact Assessment standard."""
    return ISOStandard(
        code="ISO/IEC 42005:2025",
        name="AI Impact Assessment",
        version="2025",
        thresholds=[
            ISOThreshold(
                metric="risk_level",
                compliant=1.0,  # LOW
                caution=2.0,  # MEDIUM
                critical=3.0,  # HIGH
                direction="lower_is_better",
            ),
        ],
        mitigations=[
            MitigationStrategy(
                name="Human-in-the-loop",
                phase="post-processing",
                description="Maintenir supervision humaine sur decisions",
                effectiveness="high",
                implementation="Advisory-only predictions, human final decision",
            ),
            MitigationStrategy(
                name="Drift Monitoring",
                phase="post-processing",
                description="Surveiller la derive des donnees et performances",
                effectiveness="medium",
                implementation="PSI monitoring with alerting",
            ),
        ],
        reference_url="https://www.iso.org/standard/81283.html",
    )
