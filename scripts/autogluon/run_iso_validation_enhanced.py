"""Runner: ISO Validation Améliorée - 24027/24029/42005.

Document ID: ALICE-SCRIPT-ISORUN-002
Version: 2.0.0
ISO: 24027, 24029, 42005, 5055 (<50 lignes)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.autogluon.iso_types import ISO24027EnhancedReport, ISO24029EnhancedReport


def main() -> int:
    """Exécute les validations ISO améliorées."""
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    from scripts.autogluon.iso_fairness_enhanced import validate_fairness_enhanced
    from scripts.autogluon.iso_impact_assessment_enhanced import (
        AssessmentPhase,
        assess_impact_enhanced,
        save_report,
    )
    from scripts.autogluon.iso_robustness_enhanced import validate_robustness_enhanced

    # Chemins
    model_path = Path("models/autogluon/autogluon_20260117_042708")
    test_path = Path("data/features/test.parquet")
    reports_dir = Path("reports")

    # Fallback: trouver le modèle le plus récent
    if not model_path.exists():
        autogluon_dir = Path("models/autogluon")
        if autogluon_dir.exists():
            candidates = sorted(autogluon_dir.glob("autogluon_*"), reverse=True)
            for c in candidates:
                if (c / "predictor.pkl").exists():
                    model_path = c
                    break

    print("=" * 60)
    print("ISO VALIDATION AMÉLIORÉE - ALICE Engine")
    print("=" * 60)

    # Charger modèle et données
    print("\n[1/4] Chargement modèle et données...")
    predictor = TabularPredictor.load(str(model_path))
    test_data = pd.read_parquet(test_path)

    # Renommer la colonne cible si nécessaire
    label = predictor.label
    if label not in test_data.columns and "resultat_blanc" in test_data.columns:
        test_data = test_data.rename(columns={"resultat_blanc": label})
        print(f"  - Renamed 'resultat_blanc' -> '{label}'")

    print(f"  - Modèle: {model_path}")
    print(f"  - Test: {len(test_data)} échantillons")

    # ISO 24027 - Fairness Enhanced
    print("\n[2/4] ISO 24027 - Validation Fairness Améliorée...")
    fairness = validate_fairness_enhanced(predictor, test_data, "ligue_code")
    _save_fairness_report(fairness, reports_dir / "iso24027_fairness_enhanced.json")
    print(f"  - Demographic Parity: {fairness.metrics.demographic_parity_ratio:.2%}")
    print(f"  - Equalized Odds: {fairness.metrics.equalized_odds_ratio:.2%}")
    print(f"  - Status: {fairness.status}")
    print(f"  - Root Cause: {fairness.root_cause[:80]}...")

    # ISO 24029 - Robustness Enhanced
    print("\n[3/4] ISO 24029 - Validation Robustesse Améliorée...")
    robustness = validate_robustness_enhanced(predictor, test_data)
    _save_robustness_report(robustness, reports_dir / "iso24029_robustness_enhanced.json")
    print(f"  - Noise Tolerance: {robustness.overall_noise_tolerance:.2%}")
    print(f"  - Stability Score: {robustness.overall_stability_score:.2%}")
    print(f"  - Consistency: {robustness.consistency_test.consistency_rate:.2%}")
    print(f"  - Status: {robustness.status}")

    # ISO 42005 - Impact Assessment Enhanced
    print("\n[4/4] ISO 42005 - Impact Assessment Ameliore...")

    # Sauvegarder d'abord les rapports fairness/robustness
    _save_fairness_report(fairness, reports_dir / "iso24027_fairness_enhanced.json")
    _save_robustness_report(robustness, reports_dir / "iso24029_robustness_enhanced.json")

    impact = assess_impact_enhanced(
        reports_dir / "iso24027_fairness_enhanced.json",
        reports_dir / "iso24029_robustness_enhanced.json",
        reports_dir / "iso42001_model_card.json",
        phase=AssessmentPhase.OPERATIONAL,
    )
    save_report(impact, reports_dir / "iso42005_impact_assessment_enhanced.json")
    print(f"  - Overall Impact: {impact.overall_impact_level.value}")
    print(f"  - Recommendation: {impact.recommendation}")
    print(f"  - Next Review: {impact.next_assessment_date}")

    # Résumé
    print("\n" + "=" * 60)
    print("RESUME CONFORMITE ISO")
    print("=" * 60)
    fair_mark = "[OK]" if fairness.compliant else "[WARN]"
    robust_mark = "[OK]" if robustness.compliant else "[WARN]"
    print(f"ISO 24027 Fairness:    {fairness.status} {fair_mark}")
    print(f"ISO 24029 Robustness:  {robustness.status} {robust_mark}")
    print(f"ISO 42005 Impact:      {impact.recommendation}")
    print("=" * 60)

    return 0 if fairness.compliant and robustness.compliant else 1


def _save_fairness_report(report: ISO24027EnhancedReport, path: Path) -> None:
    """Sauvegarde le rapport fairness."""
    from dataclasses import asdict

    data = asdict(report)
    path.write_text(json.dumps(data, indent=2, default=str))


def _save_robustness_report(report: ISO24029EnhancedReport, path: Path) -> None:
    """Sauvegarde le rapport robustness."""
    from dataclasses import asdict

    data = asdict(report)
    path.write_text(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
