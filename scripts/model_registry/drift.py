"""Drift Monitoring - ISO 5259/42001.

Ce module implémente la détection de drift:
- Population Stability Index (PSI)
- Métriques de drift ELO
- Sauvegarde/chargement rapports

Conformité ISO/IEC 5259 (Data Quality), ISO/IEC 42001 (AI Lifecycle).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.model_registry.drift_dataclasses import (
    ACCURACY_DROP_THRESHOLD,
    ELO_SHIFT_THRESHOLD,
    PSI_THRESHOLD_CRITICAL,
    PSI_THRESHOLD_WARNING,
    DriftMetrics,
    DriftReport,
)

logger = logging.getLogger(__name__)

# Re-export pour compatibilité
__all__ = [
    "PSI_THRESHOLD_WARNING",
    "PSI_THRESHOLD_CRITICAL",
    "ACCURACY_DROP_THRESHOLD",
    "ELO_SHIFT_THRESHOLD",
    "DriftMetrics",
    "DriftReport",
    "compute_psi",
    "compute_drift_metrics",
    "create_drift_report",
    "add_round_to_drift_report",
    "save_drift_report",
    "load_drift_report",
    "check_drift_status",
]


def compute_psi(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Calcule le Population Stability Index (PSI).

    PSI mesure le changement de distribution entre baseline et current.
    - PSI < 0.1: Pas de changement significatif
    - 0.1 <= PSI < 0.25: Changement modéré (warning)
    - PSI >= 0.25: Changement significatif (action requise)

    Args:
    ----
        baseline: Distribution de référence (training)
        current: Distribution actuelle (inference)
        bins: Nombre de bins pour l'histogramme

    Returns:
    -------
        Score PSI
    """
    import numpy as np

    min_val = min(baseline.min(), current.min())
    max_val = max(baseline.max(), current.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    baseline_pct = (baseline_counts + 1) / (len(baseline) + bins)
    current_pct = (current_counts + 1) / (len(current) + bins)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

    return float(psi)


def compute_drift_metrics(
    round_number: int,
    predictions: pd.DataFrame,
    actuals: pd.Series,
    baseline_elo_mean: float,
    baseline_elo_std: float,
    baseline_elo_distribution: pd.Series,
) -> DriftMetrics:
    """Calcule les métriques de drift pour une ronde.

    Args:
    ----
        round_number: Numéro de la ronde (1-9)
        predictions: DataFrame avec colonnes 'predicted_proba', 'elo_blanc', 'elo_noir'
        actuals: Série des résultats réels (0/1)
        baseline_elo_mean: Moyenne ELO du training set
        baseline_elo_std: Écart-type ELO du training set
        baseline_elo_distribution: Distribution ELO complète pour PSI

    Returns:
    -------
        DriftMetrics pour cette ronde
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    timestamp = datetime.now().isoformat()
    alerts: list[str] = []

    required_cols = {"predicted_proba", "elo_blanc", "elo_noir"}
    missing_cols = required_cols - set(predictions.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in predictions: {missing_cols}")

    predicted_classes = (predictions["predicted_proba"] >= 0.5).astype(int)
    accuracy = accuracy_score(actuals, predicted_classes)

    try:
        auc = roc_auc_score(actuals, predictions["predicted_proba"])
    except ValueError:
        auc = None

    current_elo = pd.concat([predictions["elo_blanc"], predictions["elo_noir"]])
    current_mean = current_elo.mean()
    current_std = current_elo.std()

    elo_mean_shift = abs(current_mean - baseline_elo_mean)
    elo_std_shift = abs(current_std - baseline_elo_std)

    psi = compute_psi(baseline_elo_distribution, current_elo)

    has_warning = False
    has_critical = False

    if psi >= PSI_THRESHOLD_CRITICAL:
        has_critical = True
        alerts.append(f"CRITICAL: PSI={psi:.3f} (seuil={PSI_THRESHOLD_CRITICAL})")
    elif psi >= PSI_THRESHOLD_WARNING:
        has_warning = True
        alerts.append(f"WARNING: PSI={psi:.3f} (seuil={PSI_THRESHOLD_WARNING})")

    if elo_mean_shift > ELO_SHIFT_THRESHOLD:
        has_warning = True
        alerts.append(f"WARNING: ELO mean shift={elo_mean_shift:.1f} points")

    return DriftMetrics(
        round_number=round_number,
        timestamp=timestamp,
        predictions_count=len(predictions),
        accuracy=accuracy,
        auc_roc=auc,
        elo_mean_shift=elo_mean_shift,
        elo_std_shift=elo_std_shift,
        psi_score=psi,
        has_warning=has_warning,
        has_critical=has_critical,
        alerts=alerts,
    )


def create_drift_report(
    season: str,
    model_version: str,
    training_elo: pd.Series,
) -> DriftReport:
    """Crée un rapport de drift vide pour la saison."""
    return DriftReport(
        season=season,
        model_version=model_version,
        baseline_elo_mean=training_elo.mean(),
        baseline_elo_std=training_elo.std(),
        rounds=[],
    )


def add_round_to_drift_report(
    report: DriftReport,
    round_number: int,
    predictions: pd.DataFrame,
    actuals: pd.Series,
    baseline_elo_distribution: pd.Series,
) -> DriftMetrics:
    """Ajoute les métriques d'une ronde au rapport de drift."""
    metrics = compute_drift_metrics(
        round_number=round_number,
        predictions=predictions,
        actuals=actuals,
        baseline_elo_mean=report.baseline_elo_mean,
        baseline_elo_std=report.baseline_elo_std,
        baseline_elo_distribution=baseline_elo_distribution,
    )
    report.rounds.append(metrics)
    return metrics


def save_drift_report(report: DriftReport, output_path: Path) -> None:
    """Sauvegarde le rapport de drift en JSON."""
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"  Drift report saved to {output_path.name}")


def load_drift_report(input_path: Path) -> DriftReport | None:
    """Charge un rapport de drift depuis JSON."""
    if not input_path.exists():
        logger.warning(f"Drift report not found: {input_path}")
        return None

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rounds = []
    for r in data.get("rounds", []):
        rounds.append(
            DriftMetrics(
                round_number=r["round"],
                timestamp=r["timestamp"],
                predictions_count=r["predictions"]["count"],
                accuracy=r["predictions"]["accuracy"],
                auc_roc=r["predictions"]["auc_roc"],
                elo_mean_shift=r["drift"]["elo_mean_shift"],
                elo_std_shift=r["drift"]["elo_std_shift"],
                psi_score=r["drift"]["psi_score"],
                has_warning=r["status"]["has_warning"],
                has_critical=r["status"]["has_critical"],
                alerts=r["status"]["alerts"],
            )
        )

    return DriftReport(
        season=data["season"],
        model_version=data["model_version"],
        baseline_elo_mean=data["baseline"]["elo_mean"],
        baseline_elo_std=data["baseline"]["elo_std"],
        rounds=rounds,
    )


def check_drift_status(report: DriftReport) -> dict[str, object]:
    """Vérifie le statut de drift et retourne recommandation."""
    summary = report.get_summary()

    if summary.get("status") == "no_data":
        return {"status": "NO_DATA", "message": "Aucune donnée de monitoring"}

    recommendation = summary.get("recommendation", "OK")

    messages = {
        "OK": "Modèle stable, pas d'action requise",
        "MONITOR_CLOSELY": "Légère dégradation, surveiller les prochaines rondes",
        "RETRAIN_RECOMMENDED": "Drift significatif détecté, retraining recommandé",
        "RETRAIN_URGENT": "Drift critique, retraining urgent nécessaire",
    }

    return {
        "status": recommendation,
        "message": messages.get(recommendation, "Unknown"),
        "summary": summary,
    }
