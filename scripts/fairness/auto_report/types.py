"""Types pour Fairness Report Automatique - ISO 24027.

Ce module contient les types pour le rapport fairness complet:
- GroupMetrics: Metriques disaggregees par groupe
- AttributeAnalysis: Analyse d'un attribut protege
- ComprehensiveFairnessReport: Rapport global multi-attributs

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- NIST AI 100-1 MEASURE 2.11 - Fairness evaluation
- EU AI Act Art.13 - Transparency (disaggregated metrics)
- ISO/IEC 27034 - Secure Coding (Pydantic validation)
- ISO/IEC 5055:2021 - Code Quality (SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

_FairnessStatus = Literal["fair", "caution", "critical"]


class GroupMetrics(BaseModel):
    """Metriques disaggregees pour un groupe (EU AI Act Art.13).

    Attributes
    ----------
        group_name: Nom du groupe (ex: "IDF", "GM")
        sample_count: Nombre d'echantillons dans le groupe
        positive_rate: Taux de prediction positive
        tpr: True Positive Rate (recall)
        fpr: False Positive Rate
        precision: Precision du groupe
        accuracy: Accuracy du groupe
        calibration_gap: |predicted_positive_rate - actual_positive_rate|
    """

    group_name: str = Field(min_length=1)
    sample_count: int = Field(ge=0)
    positive_rate: float = Field(ge=0.0, le=1.0)
    tpr: float = Field(ge=0.0, le=1.0)
    fpr: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    calibration_gap: float = Field(default=0.0, ge=0.0, le=1.0)


class AttributeAnalysis(BaseModel):
    """Analyse fairness pour un attribut protege.

    Attributes
    ----------
        attribute_name: Nom de l'attribut analyse
        sample_count: Nombre total d'echantillons
        group_count: Nombre de groupes distincts
        demographic_parity_ratio: Ratio min/max selection rate (EEOC 80%)
        equalized_odds_tpr_diff: Max TPR difference entre groupes
        equalized_odds_fpr_diff: Max FPR difference entre groupes
        predictive_parity_diff: Max precision difference entre groupes
        min_group_accuracy: Accuracy du groupe le moins performant
        max_calibration_gap: Max ecart calibration entre groupes (NIST)
        status: fair / caution / critical
        group_details: Metriques disaggregees par groupe (EU AI Act)
        confidence_intervals: CI bootstrap par metrique (NIST AI 100-1)
    """

    attribute_name: str = Field(min_length=1)
    sample_count: int = Field(ge=0)
    group_count: int = Field(ge=0)
    demographic_parity_ratio: float = Field(ge=0.0, le=1.0)
    equalized_odds_tpr_diff: float = Field(ge=0.0, le=1.0)
    equalized_odds_fpr_diff: float = Field(ge=0.0, le=1.0)
    predictive_parity_diff: float = Field(ge=0.0, le=1.0)
    min_group_accuracy: float = Field(ge=0.0, le=1.0)
    max_calibration_gap: float = Field(default=0.0, ge=0.0, le=1.0)
    status: _FairnessStatus = "fair"
    group_details: list[GroupMetrics] = Field(default_factory=list)
    confidence_intervals: dict[str, list[float]] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return self.model_dump()


class ComprehensiveFairnessReport(BaseModel):
    """Rapport fairness complet multi-attributs.

    Conforme ISO 24027 + NIST AI 100-1 + EU AI Act Art.13.
    """

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    total_samples: int = Field(ge=0)
    analyses: list[AttributeAnalysis] = Field(default_factory=list)
    overall_status: _FairnessStatus = "fair"
    recommendations: list[str] = Field(default_factory=list)
    iso_compliance: dict[str, bool] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour serialisation JSON."""
        return self.model_dump()
