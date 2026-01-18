"""Script: run_mcnemar.py - McNemar Comparison Runner.

Document ID: ALICE-SCRIPT-MCNEMAR-001
Version: 1.1.0
ISO: 5055 (<50 lignes/fonction), 24029 (validation statistique)
"""

from __future__ import annotations

import json
from pathlib import Path

import catboost
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.comparison.mcnemar_test import mcnemar_simple_test

FEATURES = [
    "blanc_elo",
    "noir_elo",
    "diff_elo",
    "echiquier",
    "niveau",
    "ronde",
    "type_competition",
    "division",
    "ligue_code",
    "blanc_titre",
    "noir_titre",
    "jour_semaine",
]


def main() -> None:
    """Execute McNemar comparison AutoGluon vs Baseline (ISO 24029)."""
    from autogluon.tabular import TabularPredictor

    ag_model = TabularPredictor.load("models/autogluon/autogluon_extreme_v2")
    bl_model = catboost.CatBoostClassifier()
    bl_model.load_model(str(sorted(Path("models").glob("v*/catboost.cbm"))[-1]))

    test_df = pd.read_parquet("data/features/test.parquet")
    test_df["target"] = (test_df["resultat_blanc"] == 1.0).astype(int)
    features_df, labels = test_df[FEATURES], test_df["target"].values

    auc_ag = roc_auc_score(labels, ag_model.predict_proba(features_df)[1])
    auc_bl = roc_auc_score(labels, bl_model.predict_proba(features_df.values)[:, 1])

    mcnemar = mcnemar_simple_test(
        labels, ag_model.predict(features_df).values, bl_model.predict(features_df.values)
    )
    diff_pct = (auc_ag - auc_bl) * 100

    Path("reports/mcnemar_comparison.json").write_text(
        json.dumps(
            {
                "autogluon_auc": auc_ag,
                "baseline_auc": auc_bl,
                "difference_pct": diff_pct,
                "p_value": mcnemar.p_value,
                "significant": mcnemar.significant,
                "meets_2pct": diff_pct >= 2.0,
                "winner": "AutoGluon" if mcnemar.significant and diff_pct >= 2.0 else "Baseline",
            },
            indent=2,
        )
    )
    print(f"AG:{auc_ag:.4f} BL:{auc_bl:.4f} diff:{diff_pct:+.2f}% p:{mcnemar.p_value:.4f}")


if __name__ == "__main__":
    main()
