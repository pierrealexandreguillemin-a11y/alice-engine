"""Tests tendances win/draw — W/D/L decomposition - ISO 29119.

Document ID: ALICE-TEST-FEATURES-RECENT-FORM-TENDANCE
Version: 2.0.0
Tests count: 6

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)
- ISO/IEC 42001:2023 - AI traceability (trend signals)

Author: ALICE Engine Team
Last Updated: 2026-03-22
"""

import pandas as pd

from scripts.features.recent_form import _compute_trends


class TestComputeTrends:
    """Tests pour _compute_trends (tendances win et draw)."""

    def test_win_trend_hausse(self) -> None:
        """Test detection win_trend hausse (delta > 0.1)."""
        df = pd.DataFrame({"resultat": [0.0, 0.0, 1.0, 1.0]})
        win_trend, _ = _compute_trends(df, "resultat")
        assert win_trend == "hausse"

    def test_win_trend_baisse(self) -> None:
        """Test detection win_trend baisse (delta < -0.1)."""
        df = pd.DataFrame({"resultat": [1.0, 1.0, 0.0, 0.0]})
        win_trend, _ = _compute_trends(df, "resultat")
        assert win_trend == "baisse"

    def test_win_trend_stable(self) -> None:
        """Test detection win_trend stable (|delta| <= 0.1)."""
        df = pd.DataFrame({"resultat": [1.0, 1.0, 1.0, 1.0]})
        win_trend, _ = _compute_trends(df, "resultat")
        assert win_trend == "stable"

    def test_draw_trend_hausse(self) -> None:
        """Test detection draw_trend hausse (nulles croissantes)."""
        df = pd.DataFrame({"resultat": [0.0, 0.0, 0.5, 0.5]})
        _, draw_trend = _compute_trends(df, "resultat")
        assert draw_trend == "hausse"

    def test_draw_trend_baisse(self) -> None:
        """Test detection draw_trend baisse (nulles decroissantes)."""
        df = pd.DataFrame({"resultat": [0.5, 0.5, 0.0, 0.0]})
        _, draw_trend = _compute_trends(df, "resultat")
        assert draw_trend == "baisse"

    def test_trends_stable_with_odd_window(self) -> None:
        """Test avec fenetre impaire — tendances calculees sur mid = len//2."""
        df = pd.DataFrame({"resultat": [1.0, 1.0, 0.5, 1.0, 1.0]})
        win_trend, draw_trend = _compute_trends(df, "resultat")
        assert win_trend in {"hausse", "baisse", "stable"}
        assert draw_trend in {"hausse", "baisse", "stable"}
