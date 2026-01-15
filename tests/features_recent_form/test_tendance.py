"""Tests Calculate Tendance - ISO 29119.

Document ID: ALICE-TEST-FEATURES-RECENT-FORM-TENDANCE
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.recent_form import _calculate_tendance


class TestCalculateTendance:
    """Tests pour _calculate_tendance (fonction interne)."""

    def test_tendance_hausse(self) -> None:
        """Test detection tendance hausse (delta > 0.1)."""
        df = pd.DataFrame({"resultat": [0.0, 0.0, 1.0, 1.0]})
        result = _calculate_tendance(df, "resultat", 4)
        assert result == "hausse"

    def test_tendance_baisse(self) -> None:
        """Test detection tendance baisse (delta < -0.1)."""
        df = pd.DataFrame({"resultat": [1.0, 1.0, 0.0, 0.0]})
        result = _calculate_tendance(df, "resultat", 4)
        assert result == "baisse"

    def test_tendance_stable(self) -> None:
        """Test detection tendance stable (|delta| <= 0.1)."""
        df = pd.DataFrame({"resultat": [0.5, 0.5, 0.5, 0.5]})
        result = _calculate_tendance(df, "resultat", 4)
        assert result == "stable"

    def test_tendance_boundary_hausse(self) -> None:
        """Test limite hausse exactement a 0.1 (devrait etre stable)."""
        df = pd.DataFrame({"resultat": [0.4, 0.5, 0.5, 0.6]})
        result = _calculate_tendance(df, "resultat", 4)
        assert result == "stable"

    def test_tendance_boundary_above_threshold(self) -> None:
        """Test juste au-dessus du seuil 0.1."""
        df = pd.DataFrame({"resultat": [0.0, 0.0, 0.2, 0.2]})
        result = _calculate_tendance(df, "resultat", 4)
        assert result == "hausse"

    def test_tendance_with_odd_window(self) -> None:
        """Test avec fenetre impaire."""
        df = pd.DataFrame({"resultat": [0.0, 0.0, 0.5, 1.0, 1.0]})
        result = _calculate_tendance(df, "resultat", 5)
        assert result == "hausse"
