"""Script: generate_iso25059.py - ISO 25059 Report Generator.

Document ID: ALICE-SCRIPT-ISO25059-001
Version: 1.1.0
ISO: 5055 (<50 lignes/fonction), 25059 (AI Quality)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _status(condition: bool) -> str:
    """Return PASS/FAIL status string."""
    return "PASS" if condition else "FAIL"


def main() -> None:
    """Generate ISO 25059 training report."""
    ag = json.loads(Path("reports/autogluon_results.json").read_text())
    rob = json.loads(Path("reports/robustness_report.json").read_text())
    fair = json.loads(Path("reports/fairness_report.json").read_text())
    mcn = json.loads(Path("reports/mcnemar_comparison.json").read_text())
    bl = json.loads(sorted(Path("models").glob("v*/metadata.json"))[-1].read_text())

    report = f"""# Rapport ISO 25059 v2.0 - {datetime.now():%Y-%m-%d %H:%M}

## Resume
| Critere | Resultat | Statut |
|---------|----------|--------|
| AutoGluon AUC | {ag["test_auc"]:.4f} | {_status(ag["test_auc"] >= 0.70)} |
| ISO 24029 | {rob["status"]} | {_status(rob["status"] != "FRAGILE")} |
| ISO 24027 | {fair["status"]} | {_status(fair["status"] != "CRITICAL")} |
| Diff vs Baseline | {mcn["difference_pct"]:+.2f}% | {_status(mcn["meets_2pct"])} |
| p-value | {mcn["p_value"]:.4f} | {_status(mcn["significant"])} |

**Recommandation:** {mcn["winner"]}

## Baseline
| Model | Test AUC |
|-------|----------|
| CatBoost | {bl["metrics"]["CatBoost"]["test_auc"]:.4f} |
| XGBoost | {bl["metrics"]["XGBoost"]["test_auc"]:.4f} |
| LightGBM | {bl["metrics"]["LightGBM"]["test_auc"]:.4f} |

## Decision
-> **{mcn["winner"]}**
"""
    Path("reports/ISO_25059_TRAINING_REPORT_v2.md").write_text(report)
    print("Generated: reports/ISO_25059_TRAINING_REPORT_v2.md")


if __name__ == "__main__":
    main()
