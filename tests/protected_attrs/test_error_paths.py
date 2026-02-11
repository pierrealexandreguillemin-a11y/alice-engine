"""Tests Error Paths - Protected Attributes - ISO 29119.

Document ID: ALICE-TEST-PROTECTED-ERROR-PATHS
Version: 1.0.0
Tests count: 11

Covers:
- Empty features list
- NaN values in DataFrame
- Protected attr not in DataFrame columns
- Single-value column for Cramer's V
- Empty DataFrame for proxy detection
- Pydantic validators on types
- Nominal protected attr uses Cramer's V (not Pearson)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing (error paths)
- ISO/IEC TR 24027:2021 - Bias in AI systems

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from scripts.fairness.protected.types import (
    ProtectedAttribute,
    ProtectionLevel,
    ProxyCorrelation,
)
from scripts.fairness.protected.validator import (
    _cramers_v,
    detect_proxy_correlations,
    validate_features,
)


class TestEmptyInputs:
    """Tests pour les entrees vides."""

    def test_empty_features_list_is_valid(self) -> None:
        """Liste de features vide -> validation passe."""
        protected = [
            ProtectedAttribute(
                name="gender",
                level=ProtectionLevel.FORBIDDEN,
                reason="test",
            ),
        ]
        result = validate_features([], protected)
        assert result.is_valid is True
        assert len(result.violations) == 0

    def test_empty_protected_list(self) -> None:
        """Liste d'attributs proteges vide -> validation passe."""
        result = validate_features(["blanc_elo"], [])
        assert result.is_valid is True

    def test_empty_dataframe_no_proxies(self) -> None:
        """DataFrame vide -> pas de proxies detectes."""
        df = pd.DataFrame()
        protected = [
            ProtectedAttribute(
                name="gender",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        correlations = detect_proxy_correlations(df, ["feature"], protected)
        assert len(correlations) == 0


class TestNaNHandling:
    """Tests pour les valeurs NaN."""

    def test_all_nan_column_no_crash(self) -> None:
        """Colonne entierement NaN ne crash pas."""
        n = 100
        df = pd.DataFrame(
            {
                "feature": [np.nan] * n,
                "protected": np.random.default_rng(42).standard_normal(n),
            }
        )
        protected = [
            ProtectedAttribute(
                name="protected",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        # Should not raise - NaN corrcoef returns NaN -> 0.0
        correlations = detect_proxy_correlations(df, ["feature"], protected)
        # Either empty or with 0.0 correlation (below threshold)
        assert all(c.correlation < 0.7 for c in correlations) or len(correlations) == 0


class TestProtectedAttrNotInDF:
    """Tests pour les attributs absents du DataFrame."""

    def test_protected_not_in_columns_skipped(self) -> None:
        """Attribut protege absent des colonnes est ignore."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "feature_a": rng.integers(0, 10, 100),
            }
        )
        protected = [
            ProtectedAttribute(
                name="nonexistent",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        correlations = detect_proxy_correlations(df, ["feature_a"], protected)
        assert len(correlations) == 0

    def test_feature_not_in_columns_skipped(self) -> None:
        """Feature absente des colonnes est ignoree."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "protected": rng.choice(["A", "B"], 100),
            }
        )
        protected = [
            ProtectedAttribute(
                name="protected",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        correlations = detect_proxy_correlations(df, ["nonexistent_feature"], protected)
        assert len(correlations) == 0


class TestNominalProtectedAttr:
    """Tests pour les attributs proteges nominaux (non-numeriques)."""

    def test_nominal_protected_uses_cramers_v(self) -> None:
        """Attribut protege nominal utilise Cramer's V (pas Pearson)."""
        rng = np.random.default_rng(42)
        n = 500
        # Feature numerique, protected nominal
        groups = rng.choice(["M", "F"], n)
        feature = np.where(groups == "M", 1.0, 0.0) + rng.normal(0, 0.01, n)
        df = pd.DataFrame({"num_feature": feature, "gender": groups})
        protected = [
            ProtectedAttribute(
                name="gender",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        correlations = detect_proxy_correlations(df, ["num_feature"], protected)
        assert len(correlations) >= 1
        assert correlations[0].method == "cramers_v"


class TestCramersVEdgeCases:
    """Tests pour les cas limites de Cramer's V."""

    def test_two_elements_only(self) -> None:
        """Deux elements seulement ne crash pas."""
        x = np.array(["A", "B"])
        y = np.array(["X", "Y"])
        v = _cramers_v(x, y)
        assert 0.0 <= v <= 1.0

    def test_all_same_values_both_columns(self) -> None:
        """Les deux colonnes identiques et constantes -> 0."""
        x = np.array(["A"] * 100)
        y = np.array(["X"] * 100)
        v = _cramers_v(x, y)
        assert v == 0.0


class TestTypesValidation:
    """Tests pour la validation Pydantic des types."""

    def test_proxy_correlation_method_literal(self) -> None:
        """ProxyCorrelation n'accepte que pearson/cramers_v."""
        with pytest.raises(ValidationError):
            ProxyCorrelation(
                feature="test",
                protected_attr="gender",
                correlation=0.8,
                method="invalid_method",
            )

    def test_protected_attribute_empty_name_rejected(self) -> None:
        """ProtectedAttribute avec nom vide est rejete."""
        with pytest.raises(ValidationError):
            ProtectedAttribute(
                name="",
                level=ProtectionLevel.FORBIDDEN,
                reason="test",
            )
