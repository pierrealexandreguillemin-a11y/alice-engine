"""Module: scripts/autogluon/iso_compliance.py - ISO Compliance Validation.

Document ID: ALICE-MOD-AUTOGLUON-ISO-001
Version: 1.0.0

Verification de conformite ISO 42001/24029/24027 pour AutoGluon.
Genere Model Cards, rapports de robustesse et de biais.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Model Card)
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC TR 24027:2021 - Bias Detection
- ISO/IEC 5055:2021 - Code Quality (<250 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor

    from scripts.autogluon.trainer import AutoGluonTrainingResult

logger = logging.getLogger(__name__)


@dataclass
class ISO42001ModelCard:
    """Model Card conforme ISO 42001.

    Documentation complete du modele pour tracabilite et gouvernance AI.

    Attributes
    ----------
        model_id: Identifiant unique du modele
        model_name: Nom du modele
        version: Version du modele
        created_at: Date de creation
        training_data_hash: Hash des donnees d'entrainement
        hyperparameters: Hyperparametres utilises
        metrics: Metriques de performance
        intended_use: Cas d'usage prevu
        limitations: Limitations connues
        ethical_considerations: Considerations ethiques
    """

    model_id: str
    model_name: str
    version: str
    created_at: str
    training_data_hash: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    intended_use: str = "Prediction de resultats d'echecs (classification binaire)"
    limitations: str = "Donnees FFE France uniquement, competitions officielles"
    ethical_considerations: str = "Pas de donnees personnelles sensibles"
    best_model: str = ""
    num_models_trained: int = 0
    feature_importance: dict[str, float] = field(default_factory=dict)


@dataclass
class ISO24029RobustnessReport:
    """Rapport de robustesse conforme ISO 24029.

    Analyse de la robustesse du modele face aux perturbations.
    """

    model_id: str
    noise_tolerance: float  # Tolerance au bruit (% accuracy maintenue)
    adversarial_robustness: float  # Robustesse adversariale
    distribution_shift_score: float  # Score face au shift de distribution
    confidence_calibration: float  # Calibration des confiances
    status: str = "NOT_EVALUATED"  # ROBUST, SENSITIVE, FRAGILE, NOT_EVALUATED


@dataclass
class ISO24027BiasReport:
    """Rapport de biais conforme ISO 24027.

    Analyse de l'equite du modele.
    """

    model_id: str
    demographic_parity: float  # Parite demographique
    equalized_odds: float  # Odds equalises
    calibration_by_group: dict[str, float] = field(default_factory=dict)
    status: str = "NOT_EVALUATED"  # FAIR, CAUTION, CRITICAL, NOT_EVALUATED


def generate_model_card(
    result: AutoGluonTrainingResult,
    output_path: Path | None = None,
) -> ISO42001ModelCard:
    """Genere une Model Card ISO 42001 pour le modele AutoGluon.

    Args:
    ----
        result: Resultat d'entrainement AutoGluon
        output_path: Chemin de sauvegarde (optionnel)

    Returns:
    -------
        ISO42001ModelCard complete

    ISO 42001: Documentation obligatoire du modele.
    """
    # Extraire l'importance des features si disponible
    feature_importance = {}
    try:
        fi_df = result.predictor.feature_importance()
        feature_importance = fi_df["importance"].to_dict()
    except Exception:
        logger.debug("Feature importance not available")

    model_card = ISO42001ModelCard(
        model_id=str(result.model_path.name),
        model_name=f"AutoGluon_{result.config.presets}",
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        training_data_hash=result.data_hash,
        hyperparameters={
            "presets": result.config.presets,
            "time_limit": result.config.time_limit,
            "eval_metric": result.config.eval_metric,
            "num_bag_folds": result.config.num_bag_folds,
            "num_stack_levels": result.config.num_stack_levels,
        },
        metrics=result.metrics,
        best_model=result.best_model,
        num_models_trained=int(result.metrics.get("num_models", 0)),
        feature_importance=feature_importance,
    )

    # Sauvegarder si chemin specifie
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(model_card), f, indent=2, default=str)
        logger.info(f"Model card saved to {output_path}")

    return model_card


def validate_robustness(
    predictor: TabularPredictor,
    test_data: Any,
    noise_level: float = 0.1,
) -> ISO24029RobustnessReport:
    """Valide la robustesse du modele selon ISO 24029.

    Args:
    ----
        predictor: TabularPredictor a evaluer
        test_data: Donnees de test
        noise_level: Niveau de bruit a appliquer (defaut 10%)

    Returns:
    -------
        ISO24029RobustnessReport

    ISO 24029: Tests de robustesse obligatoires.
    """
    import numpy as np

    model_id = str(predictor.path)

    # Evaluation baseline
    label = predictor.label
    y_true = test_data[label]
    X_test = test_data.drop(columns=[label])
    y_pred_baseline = predictor.predict(X_test)
    baseline_acc = (y_pred_baseline == y_true).mean()

    # Test avec bruit gaussien sur features numeriques
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns
    X_noisy = X_test.copy()

    for col in numeric_cols:
        noise = np.random.normal(0, noise_level * X_noisy[col].std(), len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise

    y_pred_noisy = predictor.predict(X_noisy)
    noisy_acc = (y_pred_noisy == y_true).mean()

    # Calculer les metriques
    noise_tolerance = noisy_acc / baseline_acc if baseline_acc > 0 else 0

    # Determiner le statut
    if noise_tolerance >= 0.95:
        status = "ROBUST"
    elif noise_tolerance >= 0.85:
        status = "SENSITIVE"
    else:
        status = "FRAGILE"

    return ISO24029RobustnessReport(
        model_id=model_id,
        noise_tolerance=float(noise_tolerance),
        adversarial_robustness=0.0,  # Requires specialized testing
        distribution_shift_score=0.0,  # Requires separate test data
        confidence_calibration=0.0,  # Requires probability analysis
        status=status,
    )


def validate_fairness(
    predictor: TabularPredictor,
    test_data: Any,
    protected_attribute: str,
) -> ISO24027BiasReport:
    """Valide l'equite du modele selon ISO 24027.

    Args:
    ----
        predictor: TabularPredictor a evaluer
        test_data: Donnees de test
        protected_attribute: Attribut protege (ex: 'ligue_code')

    Returns:
    -------
        ISO24027BiasReport

    ISO 24027: Detection de biais obligatoire.
    """
    model_id = str(predictor.path)
    label = predictor.label

    y_pred = predictor.predict(test_data.drop(columns=[label]))
    protected = test_data[protected_attribute]

    # Calculer le taux de prediction positive par groupe
    groups = protected.unique()
    positive_rates = {}

    for group in groups:
        mask = protected == group
        if mask.sum() > 0:
            positive_rates[str(group)] = float((y_pred[mask] == 1).mean())

    # Demographic parity ratio (min/max)
    if len(positive_rates) > 0:
        rates = list(positive_rates.values())
        min_rate = min(rates) if rates else 0
        max_rate = max(rates) if rates else 1
        demographic_parity = min_rate / max_rate if max_rate > 0 else 0
    else:
        demographic_parity = 1.0

    # Determiner le statut (EEOC 80% rule)
    if demographic_parity >= 0.8:
        status = "FAIR"
    elif demographic_parity >= 0.6:
        status = "CAUTION"
    else:
        status = "CRITICAL"

    return ISO24027BiasReport(
        model_id=model_id,
        demographic_parity=float(demographic_parity),
        equalized_odds=0.0,  # Requires more detailed analysis
        calibration_by_group=positive_rates,
        status=status,
    )


def validate_iso_compliance(
    result: AutoGluonTrainingResult,
    test_data: Any,
    protected_attribute: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validation complete ISO 42001/24029/24027.

    Args:
    ----
        result: Resultat d'entrainement AutoGluon
        test_data: Donnees de test
        protected_attribute: Attribut protege pour test de biais
        output_dir: Repertoire de sauvegarde des rapports

    Returns:
    -------
        Dict avec Model Card et rapports de conformite

    ISO 42001/24029/24027: Validation complete obligatoire.
    """
    output_dir = output_dir or result.model_path
    output_dir = Path(output_dir)

    # Model Card (ISO 42001)
    model_card = generate_model_card(
        result,
        output_path=output_dir / "model_card.json",
    )

    # Robustesse (ISO 24029)
    robustness = validate_robustness(result.predictor, test_data)

    # Biais (ISO 24027) - si attribut protege specifie
    fairness = None
    if protected_attribute:
        fairness = validate_fairness(result.predictor, test_data, protected_attribute)

    # Sauvegarder les rapports
    with open(output_dir / "robustness_report.json", "w", encoding="utf-8") as f:
        json.dump(asdict(robustness), f, indent=2)

    if fairness:
        with open(output_dir / "fairness_report.json", "w", encoding="utf-8") as f:
            json.dump(asdict(fairness), f, indent=2)

    logger.info(f"ISO compliance reports saved to {output_dir}")

    return {
        "model_card": model_card,
        "robustness": robustness,
        "fairness": fairness,
        "compliant": robustness.status != "FRAGILE"
        and (fairness is None or fairness.status != "CRITICAL"),
    }
