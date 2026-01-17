"""Script: generate_iso25059.py - ISO 25059 Report Generator.

Document ID: ALICE-SCRIPT-ISO25059-001
Version: 1.0.0

Runner script pour Phase 5 du plan ML ISO.
Genere le rapport final de conformite.

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (<50 lignes)
- ISO/IEC 25059:2023 - AI Quality Model

Author: ALICE Engine Team
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    """Generate ISO 25059 training report."""
    autogluon = json.loads(Path("reports/autogluon_results.json").read_text())
    robustness = json.loads(Path("reports/robustness_report.json").read_text())
    fairness = json.loads(Path("reports/fairness_report.json").read_text())
    mcnemar = json.loads(Path("reports/mcnemar_comparison.json").read_text())
    baseline = json.loads(sorted(Path("models").glob("v*/metadata.json"))[-1].read_text())

    ok = lambda cond: "PASS" if cond else "FAIL"  # noqa: E731

    report = f"""# Rapport ISO 25059 v2.0 - {datetime.now():%Y-%m-%d %H:%M}

## Resume
| Critere | Resultat | Statut |
|---------|----------|--------|
| AutoGluon AUC | {autogluon['test_auc']:.4f} | {ok(autogluon['test_auc'] >= 0.70)} |
| ISO 24029 | {robustness['status']} | {ok(robustness['status'] != 'FRAGILE')} |
| ISO 24027 | {fairness['status']} | {ok(fairness['status'] != 'CRITICAL')} |
| Diff vs Baseline | {mcnemar['difference_pct']:+.2f}% | {ok(mcnemar['meets_2pct'])} |
| p-value | {mcnemar['p_value']:.4f} | {ok(mcnemar['significant'])} |

**Recommandation:** {mcnemar['winner']}

## Baseline
| Model | Test AUC |
|-------|----------|
| CatBoost | {baseline['metrics']['CatBoost']['test_auc']:.4f} |
| XGBoost | {baseline['metrics']['XGBoost']['test_auc']:.4f} |
| LightGBM | {baseline['metrics']['LightGBM']['test_auc']:.4f} |

## AutoGluon ({autogluon['num_models']} models)
Best: {autogluon['best_model']}

## Decision
Regle: AutoGluon si AUC>=0.70 ET !FRAGILE ET !CRITICAL ET p<0.05 ET diff>=+2%
-> **{mcnemar['winner']}**
"""
    Path("reports/ISO_25059_TRAINING_REPORT_v2.md").write_text(report)
    print("Generated: reports/ISO_25059_TRAINING_REPORT_v2.md")


if __name__ == "__main__":
    main()
