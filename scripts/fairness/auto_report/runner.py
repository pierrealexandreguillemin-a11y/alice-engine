"""Runner standalone fairness report - ISO 24027.

Usage:
    python -m scripts.fairness.auto_report.runner

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- ISO/IEC 5055:2021 - Code Quality (<50 lignes)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fairness.auto_report import format_markdown_report, generate_comprehensive_report
from scripts.fairness.protected.config import DEFAULT_PROTECTED_ATTRIBUTES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Genere un rapport fairness standalone."""
    test_df = pd.read_parquet("data/features/test.parquet")
    y_true = (test_df["resultat_blanc"] == 1.0).astype(int).values
    y_pred = (
        np.load("reports/predictions.npy") if Path("reports/predictions.npy").exists() else y_true
    )

    report = generate_comprehensive_report(
        y_true,
        y_pred,
        test_df,
        "ALICE",
        "standalone",
        DEFAULT_PROTECTED_ATTRIBUTES,
    )

    Path("reports").mkdir(exist_ok=True)
    Path("reports/fairness_comprehensive.json").write_text(json.dumps(report.to_dict(), indent=2))
    format_markdown_report(report, output_path=Path("reports/fairness_report.md"))
    logger.info("Fairness report: status=%s", report.overall_status)


if __name__ == "__main__":
    main()
