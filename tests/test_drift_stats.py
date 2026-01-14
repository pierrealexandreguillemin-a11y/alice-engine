"""Tests Statistics Drift - ISO 23894.

Document ID: ALICE-TEST-DRIFT-STATS
Version: 1.0.0
Tests: 8

Classes:
- TestComputePSI: Tests PSI (3 tests)
- TestComputeKSTest: Tests KS-test (2 tests)
- TestComputeChi2Test: Tests Chi-squared (2 tests)
- TestComputeJSDivergence: Tests JS divergence (1 test)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<120 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import numpy as np

from scripts.model_registry.drift_stats import (
    compute_chi2_test,
    compute_js_divergence,
    compute_ks_test,
    compute_psi,
)

# Fixtures (numeric_baseline, numeric_shifted, numeric_similar) are auto-loaded via pytest_plugins


class TestComputePSI:
    """Tests pour compute_psi."""

    def test_psi_identical(self, numeric_baseline):
        """PSI proche de 0 pour distributions identiques."""
        psi = compute_psi(numeric_baseline, numeric_baseline)
        assert psi < 0.01

    def test_psi_similar(self, numeric_baseline, numeric_similar):
        """PSI faible pour distributions similaires."""
        psi = compute_psi(numeric_baseline, numeric_similar)
        assert psi < 0.1  # PSI_THRESHOLD_OK

    def test_psi_shifted(self, numeric_baseline, numeric_shifted):
        """PSI élevé pour distributions différentes."""
        psi = compute_psi(numeric_baseline, numeric_shifted)
        assert psi > 0.2  # PSI_THRESHOLD_WARNING


class TestComputeKSTest:
    """Tests pour compute_ks_test."""

    def test_ks_similar(self, numeric_baseline, numeric_similar):
        """KS pvalue haute pour distributions similaires."""
        stat, pvalue = compute_ks_test(numeric_baseline, numeric_similar)
        assert 0 <= stat <= 1
        assert pvalue > 0.01

    def test_ks_shifted(self, numeric_baseline, numeric_shifted):
        """KS pvalue basse pour distributions différentes."""
        stat, pvalue = compute_ks_test(numeric_baseline, numeric_shifted)
        assert stat > 0.1
        assert pvalue < 0.01


class TestComputeChi2Test:
    """Tests pour compute_chi2_test (catégoriel)."""

    def test_chi2_similar(self):
        """Chi2 pvalue haute pour catégories similaires."""
        baseline = np.array(["A", "B", "C"] * 100)
        current = np.array(["A", "B", "C"] * 80)
        chi2, pvalue = compute_chi2_test(baseline, current)
        assert chi2 >= 0
        assert pvalue > 0.05

    def test_chi2_different(self):
        """Chi2 pvalue basse pour catégories différentes."""
        baseline = np.array(["A"] * 150 + ["B"] * 100 + ["C"] * 50)
        current = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 150)  # Inversé
        chi2, pvalue = compute_chi2_test(baseline, current)
        assert chi2 > 0
        assert pvalue < 0.05


class TestComputeJSDivergence:
    """Tests pour compute_js_divergence."""

    def test_js_range(self, numeric_baseline, numeric_shifted):
        """JS divergence entre 0 et 1."""
        js = compute_js_divergence(numeric_baseline, numeric_shifted)
        assert 0 <= js <= 1

    def test_js_identical(self, numeric_baseline):
        """JS proche de 0 pour distributions identiques."""
        js = compute_js_divergence(numeric_baseline, numeric_baseline)
        assert js < 0.1


class TestEdgeCases:
    """Tests cas limites."""

    def test_small_sample(self):
        """Test avec petit échantillon."""
        baseline = np.array([1, 2, 3])
        current = np.array([1, 2, 3, 4])
        stat, pvalue = compute_ks_test(baseline, current)
        assert stat >= 0
        assert 0 <= pvalue <= 1

    def test_with_nans(self):
        """Test avec valeurs NaN."""
        baseline = np.array([1, 2, np.nan, 4, 5])
        current = np.array([1, np.nan, 3, 4, 5])
        stat, pvalue = compute_ks_test(baseline, current)
        assert 0 <= stat <= 1
