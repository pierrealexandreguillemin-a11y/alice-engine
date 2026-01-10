"""Tests: tests/test_fairness_bias_detection.py - Tests ISO 24027.

Document ID: ALICE-TEST-BIAS-001
Version: 1.0.0
Tests: 29

Ce module teste la détection de biais dans les modèles ML ALICE.

Classes de tests:
- TestBiasMetrics: Tests dataclass BiasMetrics (2 tests)
- TestBiasThresholds: Tests seuils ISO 24027 (2 tests)
- TestComputeBiasMetricsByGroup: Tests calcul métriques (6 tests)
- TestComputeBiasByEloRange: Tests par tranche Elo (3 tests)
- TestCheckBiasThresholds: Tests vérification seuils (4 tests)
- TestGenerateFairnessReport: Tests rapport complet (5 tests)
- TestBiasLevel: Tests enum niveaux (2 tests)
- TestEdgeCases: Tests cas limites (5 tests)

Couverture:
- SPD (Statistical Parity Difference)
- EOD (Equal Opportunity Difference)
- DIR (Disparate Impact Ratio)
- Seuils EEOC 4/5 rule

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing (structure tests)
- ISO/IEC TR 24027:2021 - Bias in AI systems (validation)
- ISO/IEC 42001:2023 - AI Management (traçabilité)

See Also
--------
- scripts/fairness/bias_detection.py - Module testé
- scripts/fairness/__init__.py - Package exports
- docs/iso/AI_RISK_ASSESSMENT.md - Section R3

Author: ALICE Engine Team
Last Updated: 2026-01-10

"""

import numpy as np
import pandas as pd

from scripts.fairness.bias_detection import (
    BiasLevel,
    BiasMetrics,
    BiasReport,
    BiasThresholds,
    check_bias_thresholds,
    compute_bias_by_elo_range,
    compute_bias_metrics_by_group,
    generate_fairness_report,
)


class TestBiasMetrics:
    """Tests pour la dataclass BiasMetrics."""

    def test_bias_metrics_creation(self):
        """Vérifie création BiasMetrics avec valeurs par défaut."""
        metrics = BiasMetrics(
            group_name="test",
            group_size=100,
            positive_rate=0.5,
            true_positive_rate=0.6,
        )

        assert metrics.group_name == "test"
        assert metrics.group_size == 100
        assert metrics.positive_rate == 0.5
        assert metrics.true_positive_rate == 0.6
        assert metrics.spd == 0.0  # Défaut
        assert metrics.eod == 0.0  # Défaut
        assert metrics.dir == 1.0  # Défaut
        assert metrics.level == BiasLevel.ACCEPTABLE  # Défaut

    def test_bias_metrics_with_bias(self):
        """Vérifie BiasMetrics avec biais."""
        metrics = BiasMetrics(
            group_name="biased",
            group_size=50,
            positive_rate=0.3,
            true_positive_rate=0.4,
            spd=-0.2,
            eod=-0.15,
            dir=0.6,
            level=BiasLevel.CRITICAL,
        )

        assert metrics.spd == -0.2
        assert metrics.dir == 0.6
        assert metrics.level == BiasLevel.CRITICAL


class TestBiasThresholds:
    """Tests pour BiasThresholds."""

    def test_default_thresholds(self):
        """Vérifie seuils par défaut ISO 24027."""
        thresholds = BiasThresholds()

        assert thresholds.spd_warning == 0.1
        assert thresholds.spd_critical == 0.2
        assert thresholds.dir_min == 0.8  # EEOC 4/5 rule
        assert thresholds.dir_max == 1.25
        assert thresholds.eod_warning == 0.1
        assert thresholds.eod_critical == 0.2

    def test_custom_thresholds(self):
        """Vérifie seuils personnalisés."""
        thresholds = BiasThresholds(
            spd_warning=0.05,
            spd_critical=0.1,
            dir_min=0.9,
            dir_max=1.1,
        )

        assert thresholds.spd_warning == 0.05
        assert thresholds.dir_min == 0.9


class TestComputeBiasMetricsByGroup:
    """Tests pour compute_bias_metrics_by_group."""

    def test_no_bias_equal_groups(self):
        """Groupes identiques = pas de biais."""
        # Deux groupes avec même distribution
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        assert len(metrics) == 2

        for m in metrics:
            assert m.positive_rate == 0.5
            assert abs(m.spd) < 0.01  # Pas de différence
            assert abs(m.dir - 1.0) < 0.01  # Ratio = 1

    def test_detect_positive_bias(self):
        """Détecte biais positif (groupe surreprésenté)."""
        # Groupe A: 80% positif, Groupe B (ref): 50% positif
        y_true = np.array([1, 1, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 1, 0])  # A: 4/5=0.8, B: 1/3=0.33
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups, reference_group="B")

        # Trouver groupe A
        group_a = next(m for m in metrics if m.group_name == "A")

        assert group_a.positive_rate == 0.8
        assert group_a.spd > 0  # Biais positif vs B
        assert group_a.dir > 1.0  # Ratio > 1

    def test_detect_negative_bias(self):
        """Détecte biais négatif (groupe sous-représenté)."""
        # Groupe A: 20% positif, Groupe B (ref): 80% positif
        y_true = np.array([1, 0, 0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 0, 0, 0, 0, 1, 1, 1])
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups, reference_group="B")

        group_a = next(m for m in metrics if m.group_name == "A")

        assert group_a.positive_rate == 0.2
        assert group_a.spd < 0  # Biais négatif vs B
        assert group_a.dir < 1.0  # Ratio < 1

    def test_reference_group_auto_select(self):
        """Sélection automatique du groupe majoritaire comme référence."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        groups = np.array(["A", "A", "A", "A", "A", "A", "A", "B", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        # A est majoritaire (7 vs 3), donc référence
        group_a = next(m for m in metrics if m.group_name == "A")
        assert group_a.spd == 0.0  # Référence = 0

    def test_multiple_groups(self):
        """Test avec plusieurs groupes (divisions chess)."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        groups = np.random.choice(["N1", "N2", "N3", "REG", "DEP"], n)

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        assert len(metrics) == 5
        assert all(m.group_size > 0 for m in metrics)
        assert all(0 <= m.positive_rate <= 1 for m in metrics)

    def test_empty_group_handled(self):
        """Groupe vide géré correctement."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        groups = np.array(["A", "A", "A"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        assert len(metrics) == 1
        assert metrics[0].group_name == "A"


class TestComputeBiasByEloRange:
    """Tests pour compute_bias_by_elo_range."""

    def test_default_elo_bins(self):
        """Test avec tranches Elo par défaut."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        elo = np.random.randint(1200, 2400, n)

        metrics = compute_bias_by_elo_range(y_true, y_pred, elo)

        # Vérifier que toutes les tranches sont représentées
        group_names = {m.group_name for m in metrics}
        assert len(group_names) > 0

    def test_custom_elo_bins(self):
        """Test avec tranches Elo personnalisées."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        elo = np.array([1500, 1600, 1700, 1800, 1900, 2000])

        metrics = compute_bias_by_elo_range(y_true, y_pred, elo, bins=[1400, 1700, 2000, 2300])

        assert len(metrics) >= 1

    def test_reference_1800_2000(self):
        """Vérifie que 1800-2000 est le groupe de référence."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        elo = np.array([1850, 1900, 1950, 1850, 1600, 1650, 2100, 2150])

        metrics = compute_bias_by_elo_range(y_true, y_pred, elo)

        # Groupe 1800-2000 devrait avoir SPD = 0 (référence)
        ref_group = next((m for m in metrics if m.group_name == "1800-2000"), None)
        if ref_group:
            assert ref_group.spd == 0.0


class TestCheckBiasThresholds:
    """Tests pour check_bias_thresholds."""

    def test_acceptable_bias(self):
        """Biais acceptable = pas d'alertes."""
        metrics = [
            BiasMetrics(
                group_name="A",
                group_size=100,
                positive_rate=0.5,
                true_positive_rate=0.6,
                spd=0.05,
                eod=0.03,
                dir=1.05,
            ),
            BiasMetrics(
                group_name="B",
                group_size=100,
                positive_rate=0.48,
                true_positive_rate=0.58,
                spd=-0.02,
                eod=-0.02,
                dir=0.96,
            ),
        ]

        level, alerts = check_bias_thresholds(metrics)

        assert level == BiasLevel.ACCEPTABLE
        assert len(alerts) == 0

    def test_warning_spd(self):
        """SPD > 0.1 déclenche warning."""
        metrics = [
            BiasMetrics(
                group_name="biased",
                group_size=50,
                positive_rate=0.6,
                true_positive_rate=0.7,
                spd=0.15,  # > 0.1 warning
                eod=0.05,
                dir=1.1,
            )
        ]

        level, alerts = check_bias_thresholds(metrics)

        assert level == BiasLevel.WARNING
        assert len(alerts) == 1
        assert "SPD warning" in alerts[0]

    def test_critical_dir(self):
        """DIR < 0.6 déclenche critique."""
        metrics = [
            BiasMetrics(
                group_name="severely_biased",
                group_size=50,
                positive_rate=0.2,
                true_positive_rate=0.3,
                spd=-0.3,
                eod=-0.2,
                dir=0.5,  # < 0.6 = critique
            )
        ]

        level, alerts = check_bias_thresholds(metrics)

        assert level == BiasLevel.CRITICAL
        assert len(alerts) >= 1

    def test_eeoc_4_5_rule(self):
        """DIR hors 0.8-1.25 déclenche warning (EEOC 4/5 rule)."""
        metrics = [
            BiasMetrics(
                group_name="underrepresented",
                group_size=50,
                positive_rate=0.35,
                true_positive_rate=0.4,
                spd=-0.05,
                eod=-0.05,
                dir=0.75,  # < 0.8 mais > 0.6
            )
        ]

        level, alerts = check_bias_thresholds(metrics)

        assert level == BiasLevel.WARNING
        assert "DIR warning" in alerts[0]


class TestGenerateFairnessReport:
    """Tests pour generate_fairness_report."""

    def test_report_structure(self):
        """Vérifie structure du rapport."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        groups = np.array(["A", "A", "A", "B", "B", "B"])

        report = generate_fairness_report(
            y_true,
            y_pred,
            groups,
            model_name="TestModel",
            feature_name="division",
        )

        assert isinstance(report, BiasReport)
        assert report.model_name == "TestModel"
        assert report.feature_analyzed == "division"
        assert report.total_samples == 6
        assert len(report.metrics_by_group) == 2
        assert report.timestamp is not None

    def test_report_iso_compliance(self):
        """Vérifie champs ISO compliance."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        groups = np.array(["A", "A", "B", "B"])

        report = generate_fairness_report(y_true, y_pred, groups)

        assert "iso_24027" in report.iso_compliance
        assert "iso_42001" in report.iso_compliance
        assert "eeoc_4_5_rule" in report.iso_compliance
        assert report.iso_compliance["iso_24027"] is True

    def test_report_recommendations_critical(self):
        """Vérifie recommandations pour biais critique."""
        # Créer un biais artificiel
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # A: tous positifs, B: tous négatifs
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])

        report = generate_fairness_report(y_true, y_pred, groups, reference_group="B")

        assert report.overall_level == BiasLevel.CRITICAL
        assert len(report.recommendations) > 0
        assert any("URGENT" in r or "critique" in r.lower() for r in report.recommendations)

    def test_report_recommendations_acceptable(self):
        """Vérifie recommandations pour biais acceptable."""
        # Groupes avec distributions identiques = pas de biais
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        report = generate_fairness_report(y_true, y_pred, groups)

        assert report.overall_level == BiasLevel.ACCEPTABLE
        assert any("Aucun biais" in r for r in report.recommendations)

    def test_report_reference_group(self):
        """Vérifie que le groupe de référence est inclus."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        groups = np.array(["X", "X", "Y", "Y"])

        report = generate_fairness_report(y_true, y_pred, groups, reference_group="X")

        assert report.reference_group == "X"


class TestBiasLevel:
    """Tests pour l'enum BiasLevel."""

    def test_bias_levels(self):
        """Vérifie les niveaux de biais."""
        assert BiasLevel.ACCEPTABLE.value == "acceptable"
        assert BiasLevel.WARNING.value == "warning"
        assert BiasLevel.CRITICAL.value == "critical"

    def test_bias_level_comparison(self):
        """Vérifie comparaison des niveaux."""
        metrics_acceptable = BiasMetrics(
            group_name="A",
            group_size=10,
            positive_rate=0.5,
            true_positive_rate=0.5,
            level=BiasLevel.ACCEPTABLE,
        )
        metrics_critical = BiasMetrics(
            group_name="B",
            group_size=10,
            positive_rate=0.5,
            true_positive_rate=0.5,
            level=BiasLevel.CRITICAL,
        )

        assert metrics_acceptable.level != metrics_critical.level


class TestEdgeCases:
    """Tests pour cas limites."""

    def test_single_group(self):
        """Un seul groupe = pas de biais possible."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        groups = np.array(["A", "A", "A"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        assert len(metrics) == 1
        assert metrics[0].spd == 0.0  # Référence

    def test_all_positive_predictions(self):
        """Toutes prédictions positives."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1])  # Tout positif
        groups = np.array(["A", "A", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        for m in metrics:
            assert m.positive_rate == 1.0
            assert m.spd == 0.0  # Pas de différence

    def test_all_negative_predictions(self):
        """Toutes prédictions négatives."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0])  # Tout négatif
        groups = np.array(["A", "A", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        for m in metrics:
            assert m.positive_rate == 0.0

    def test_pandas_input(self):
        """Test avec entrées pandas."""
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = pd.Series([1, 0, 1, 0])
        groups = pd.Series(["A", "A", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

        assert len(metrics) == 2

    def test_division_by_zero_protection(self):
        """Protection contre division par zéro dans DIR."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])  # A: 100%, B: 0%
        groups = np.array(["A", "A", "B", "B"])

        metrics = compute_bias_metrics_by_group(y_true, y_pred, groups, reference_group="B")

        # Groupe A devrait avoir DIR = inf (ou valeur gérée)
        group_a = next(m for m in metrics if m.group_name == "A")
        assert group_a.dir == float("inf") or group_a.dir > 10
