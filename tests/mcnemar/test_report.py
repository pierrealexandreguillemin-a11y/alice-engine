"""Tests McNemar Report - ISO 29119.

Document ID: ALICE-TEST-MCNEMAR-REPORT
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import json
from pathlib import Path

from scripts.comparison.mcnemar_test import McNemarResult
from scripts.comparison.statistical_comparison import (
    ModelComparison,
    save_comparison_report,
)


class TestGenerateRecommendation:
    """Tests pour _generate_recommendation."""

    def test_recommendation_tie_practical(self) -> None:
        """Test recommandation pour tie avec signif pratique."""
        from scripts.comparison.statistical_comparison import _generate_recommendation

        mcnemar = McNemarResult(
            statistic=1.0,
            p_value=0.1,
            significant=False,
            effect_size=0.1,
            confidence_interval=(-0.05, 0.15),
            model_a_mean_accuracy=0.85,
            model_b_mean_accuracy=0.78,
            winner=None,
        )

        rec = _generate_recommendation(
            winner="tie",
            mcnemar=mcnemar,
            metrics_a={"accuracy": 0.85},
            metrics_b={"accuracy": 0.78},
            model_a_name="A",
            model_b_name="B",
            practical_significance=True,
        )

        assert "tendance" in rec.lower() or "A" in rec

    def test_recommendation_winner_practical(self) -> None:
        """Test recommandation pour gagnant avec signif pratique."""
        from scripts.comparison.statistical_comparison import _generate_recommendation

        mcnemar = McNemarResult(
            statistic=5.0,
            p_value=0.01,
            significant=True,
            effect_size=0.3,
            confidence_interval=(0.05, 0.15),
            model_a_mean_accuracy=0.9,
            model_b_mean_accuracy=0.7,
            winner="model_a",
        )

        rec = _generate_recommendation(
            winner="ModelA",
            mcnemar=mcnemar,
            metrics_a={"accuracy": 0.9},
            metrics_b={"accuracy": 0.7},
            model_a_name="ModelA",
            model_b_name="ModelB",
            practical_significance=True,
        )

        assert "deployer" in rec.lower() or "ModelA" in rec


class TestSaveComparisonReport:
    """Tests pour save_comparison_report."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """Test que la sauvegarde cree un fichier."""
        comparison = ModelComparison(
            model_a_name="A",
            model_b_name="B",
            mcnemar_result=McNemarResult(
                statistic=1.0,
                p_value=0.5,
                significant=False,
                effect_size=0.1,
                confidence_interval=(-0.1, 0.1),
                model_a_mean_accuracy=0.8,
                model_b_mean_accuracy=0.79,
                winner=None,
            ),
            metrics_a={"accuracy": 0.8},
            metrics_b={"accuracy": 0.79},
            winner="tie",
            practical_significance=False,
            recommendation="Choose based on operational criteria.",
        )

        output_path = tmp_path / "report.json"
        save_comparison_report(comparison, output_path)

        assert output_path.exists()

    def test_save_json_valid(self, tmp_path: Path) -> None:
        """Test que le JSON est valide."""
        comparison = ModelComparison(
            model_a_name="A",
            model_b_name="B",
            mcnemar_result=McNemarResult(
                statistic=1.0,
                p_value=0.5,
                significant=False,
                effect_size=0.1,
                confidence_interval=(-0.1, 0.1),
                model_a_mean_accuracy=0.8,
                model_b_mean_accuracy=0.79,
                winner=None,
            ),
            metrics_a={"accuracy": 0.8},
            metrics_b={"accuracy": 0.79},
            winner="tie",
            practical_significance=False,
            recommendation="Test",
        )

        output_path = tmp_path / "report.json"
        save_comparison_report(comparison, output_path)

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["winner"] == "tie"
        assert "mcnemar" in data
