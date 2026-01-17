"""Script: iso_impact_assessment.py - ISO 42005 AI Impact Assessment.

Document ID: ALICE-SCRIPT-ISO42005-001
Version: 1.0.0
ISO: 42005:2025 (AI Impact Assessment), 5055 (<50 lignes)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _assess_impact(fairness_path: Path, model_card_path: Path) -> dict:
    """Assess AI system impact per ISO 42005."""
    fair = json.loads(fairness_path.read_text()) if fairness_path.exists() else {}
    card = json.loads(model_card_path.read_text()) if model_card_path.exists() else {}

    parity = fair.get("demographic_parity", 1.0)
    group_status = "FAIR" if parity < 0.05 else "ACCEPTABLE" if parity < 0.10 else "CRITICAL"
    impact = "LOW" if parity < 0.05 else "MEDIUM" if parity < 0.10 else "HIGH"
    recommend = "REJECTED" if group_status == "CRITICAL" else "APPROVED"

    return {
        "assessment_date": datetime.now().isoformat(),
        "model_id": card.get("model_id", "unknown"),
        "iso_standard": "ISO/IEC 42005:2025",
        "impact_level": impact,
        "individual_impact": {"description": "Chess game prediction", "risk_level": "LOW"},
        "group_impact": {"demographic_parity": parity, "status": group_status},
        "societal_impact": {"description": "Team composition optimization", "risk_level": "LOW"},
        "transparency": {
            "explainability": bool(card.get("feature_importance")),
            "model_card": model_card_path.exists(),
            "data_lineage": bool(card.get("training_data_hash")),
        },
        "mitigations": ["Human-in-the-loop", "Drift monitoring", "Periodic fairness audit"],
        "recommendation": recommend,
    }


def main() -> None:
    """Generate ISO 42005 impact assessment report."""
    report = _assess_impact(
        Path("reports/fairness_report.json"),
        Path("reports/model_card_autogluon.json"),
    )
    Path("reports/impact_assessment.json").write_text(json.dumps(report, indent=2))
    print(f"ISO 42005: {report['impact_level']} impact -> {report['recommendation']}")


if __name__ == "__main__":
    main()
