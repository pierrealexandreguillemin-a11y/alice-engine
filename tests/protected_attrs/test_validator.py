"""Tests Protected Attributes Validator - ISO 29119.

Document ID: ALICE-TEST-PROTECTED-ATTRS-VALIDATOR
Version: 1.0.0
Tests count: 19

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC TR 24027:2021 - Bias in AI systems

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.fairness.protected.types import (
    ProtectedAttribute,
    ProtectionLevel,
)
from scripts.fairness.protected.validator import (
    _cramers_v,
    detect_proxy_correlations,
    validate_features,
)


class TestValidateFeaturesForbidden:
    """Tests pour la validation des attributs FORBIDDEN."""

    def test_forbidden_attr_in_features_fails(self) -> None:
        """Un attribut FORBIDDEN dans les features rend is_valid=False."""
        protected = [
            ProtectedAttribute(
                name="gender",
                level=ProtectionLevel.FORBIDDEN,
                reason="direct gender discrimination",
            ),
        ]
        features = ["blanc_elo", "noir_elo", "gender"]
        result = validate_features(features, protected)
        assert result.is_valid is False
        assert len(result.violations) >= 1
        assert any("gender" in v for v in result.violations)

    def test_forbidden_attr_not_in_features_passes(self) -> None:
        """Un attribut FORBIDDEN absent des features -> is_valid=True."""
        protected = [
            ProtectedAttribute(
                name="gender",
                level=ProtectionLevel.FORBIDDEN,
                reason="direct gender discrimination",
            ),
        ]
        features = ["blanc_elo", "noir_elo"]
        result = validate_features(features, protected)
        assert result.is_valid is True
        assert len(result.violations) == 0

    def test_multiple_forbidden_all_reported(self) -> None:
        """Tous les attributs FORBIDDEN presents sont rapportes."""
        protected = [
            ProtectedAttribute(name="gender", level=ProtectionLevel.FORBIDDEN, reason="r1"),
            ProtectedAttribute(name="ethnicity", level=ProtectionLevel.FORBIDDEN, reason="r2"),
        ]
        features = ["blanc_elo", "gender", "ethnicity"]
        result = validate_features(features, protected)
        assert result.is_valid is False
        assert len(result.violations) == 2


class TestValidateFeaturesProxyCheck:
    """Tests pour la validation des attributs PROXY_CHECK."""

    def test_proxy_attr_in_features_warns_not_fails(
        self,
        ffe_protected_attributes: list[ProtectedAttribute],
    ) -> None:
        """PROXY_CHECK genere un warning mais is_valid reste True."""
        features = ["blanc_elo", "ligue_code"]
        result = validate_features(features, ffe_protected_attributes)
        assert result.is_valid is True
        assert len(result.warnings) >= 1

    def test_is_valid_true_with_only_warnings(
        self,
        ffe_protected_attributes: list[ProtectedAttribute],
    ) -> None:
        """Avec seulement des PROXY_CHECK, is_valid=True."""
        features = ["blanc_elo", "ligue_code"]
        result = validate_features(features, ffe_protected_attributes)
        assert result.is_valid is True

    def test_warning_contains_reason(
        self,
        ffe_protected_attributes: list[ProtectedAttribute],
    ) -> None:
        """Le warning contient la raison de la protection."""
        features = ["blanc_elo", "ligue_code"]
        result = validate_features(features, ffe_protected_attributes)
        assert any("geographique" in w for w in result.warnings)


class TestDetectProxyCorrelations:
    """Tests pour la detection des correlations proxy."""

    def test_high_correlation_detected(self) -> None:
        """Correlation forte (>0.7) est detectee."""
        rng = np.random.default_rng(42)
        n = 500
        base = rng.integers(0, 3, n)
        df = pd.DataFrame(
            {
                "feature_a": base,
                "protected": base,
            }
        )
        protected = [
            ProtectedAttribute(
                name="protected",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        correlations = detect_proxy_correlations(
            df,
            ["feature_a"],
            protected,
            threshold=0.7,
        )
        assert len(correlations) >= 1
        assert correlations[0].correlation > 0.7

    def test_low_correlation_not_flagged(self) -> None:
        """Correlation faible (<0.7) n'est pas flaggee."""
        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame(
            {
                "feature_a": rng.integers(0, 100, n),
                "protected": rng.choice(["A", "B", "C"], n),
            }
        )
        protected = [
            ProtectedAttribute(
                name="protected",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        correlations = detect_proxy_correlations(
            df,
            ["feature_a"],
            protected,
            threshold=0.7,
        )
        assert len(correlations) == 0

    def test_categorical_uses_cramers_v(self) -> None:
        """Les features categorielles utilisent Cramer's V."""
        rng = np.random.default_rng(42)
        n = 500
        base = rng.choice(["X", "Y", "Z"], n)
        df = pd.DataFrame(
            {
                "cat_feature": base,
                "protected": base,
            }
        )
        protected = [
            ProtectedAttribute(
                name="protected",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        correlations = detect_proxy_correlations(
            df,
            ["cat_feature"],
            protected,
            threshold=0.7,
            categorical_features=["cat_feature"],
        )
        assert len(correlations) >= 1
        assert correlations[0].method == "cramers_v"

    def test_numeric_uses_pearson(self) -> None:
        """Les features numeriques utilisent Pearson."""
        rng = np.random.default_rng(42)
        n = 500
        values = rng.standard_normal(n)
        df = pd.DataFrame(
            {
                "num_feature": values,
                "protected": values + rng.normal(0, 0.01, n),
            }
        )
        protected = [
            ProtectedAttribute(
                name="protected",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        correlations = detect_proxy_correlations(
            df,
            ["num_feature"],
            protected,
            threshold=0.7,
        )
        assert len(correlations) >= 1
        assert correlations[0].method == "pearson"

    def test_realistic_correlation_detected(self) -> None:
        """Correlation realiste (~0.8) est detectee (pas juste 1.0)."""
        rng = np.random.default_rng(42)
        n = 500
        # Protected: 2 groups
        protected_vals = rng.choice(["A", "B"], n)
        # Feature correlated but not identical: group mean differs + noise
        feature_vals = np.where(protected_vals == "A", 5.0, 0.0)
        feature_vals = feature_vals + rng.normal(0, 1.0, n)
        df = pd.DataFrame(
            {
                "feature_a": feature_vals,
                "protected": protected_vals,
            }
        )
        protected = [
            ProtectedAttribute(
                name="protected",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        correlations = detect_proxy_correlations(
            df,
            ["feature_a"],
            protected,
            threshold=0.7,
        )
        assert len(correlations) >= 1
        # Correlation should be high but NOT 1.0
        assert 0.7 < correlations[0].correlation < 1.0

    def test_threshold_configurable(self) -> None:
        """Le seuil de correlation est configurable."""
        rng = np.random.default_rng(42)
        n = 500
        # Creer une correlation moderee (~0.3) via mapping partiel
        protected_vals = rng.choice(["A", "B"], n)
        feature_vals = np.where(protected_vals == "A", 1.0, 2.0)
        feature_vals = feature_vals + rng.normal(0, 1, n)
        df = pd.DataFrame(
            {
                "feature_a": feature_vals,
                "protected": protected_vals,
            }
        )
        protected = [
            ProtectedAttribute(
                name="protected",
                level=ProtectionLevel.PROXY_CHECK,
                reason="test",
            ),
        ]
        # Seuil bas (0.1) -> correlation moderee flaggee
        low = detect_proxy_correlations(
            df,
            ["feature_a"],
            protected,
            threshold=0.1,
        )
        # Seuil haut (0.95) -> pas flaggee
        high = detect_proxy_correlations(
            df,
            ["feature_a"],
            protected,
            threshold=0.95,
        )
        assert len(low) >= 1
        assert len(high) == 0


class TestCramersV:
    """Tests pour le calcul de Cramer's V."""

    def test_identical_columns_returns_1(self) -> None:
        """Colonnes identiques -> V = 1.0."""
        x = np.array(["A", "B", "C", "A", "B", "C"] * 50)
        y = x.copy()
        v = _cramers_v(x, y)
        assert v == pytest.approx(1.0, abs=0.01)

    def test_independent_columns_near_0(self) -> None:
        """Colonnes independantes -> V proche de 0."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.choice(["A", "B", "C"], n)
        y = rng.choice(["X", "Y", "Z"], n)
        v = _cramers_v(x, y)
        assert v < 0.15

    def test_handles_single_value_column(self) -> None:
        """Colonne avec une seule valeur -> V = 0."""
        x = np.array(["A"] * 100)
        y = np.array(["X", "Y"] * 50)
        v = _cramers_v(x, y)
        assert v == 0.0


class TestIntegration:
    """Tests d'integration bout en bout."""

    def test_full_validation_with_ffe_features(
        self,
        sample_dataframe: pd.DataFrame,
        ffe_protected_attributes: list[ProtectedAttribute],
    ) -> None:
        """Validation complete avec les features FFE (sans FORBIDDEN attrs)."""
        features = ["blanc_elo", "noir_elo", "diff_elo", "ligue_code"]
        result = validate_features(
            features,
            ffe_protected_attributes,
            df=sample_dataframe,
        )
        assert result.is_valid is True
        assert len(result.warnings) >= 1
        assert result.timestamp != ""

    def test_forbidden_attr_in_ffe_features_fails(
        self,
        sample_dataframe: pd.DataFrame,
        ffe_protected_attributes: list[ProtectedAttribute],
    ) -> None:
        """FORBIDDEN blanc_titre in features triggers violation."""
        features = ["blanc_elo", "noir_elo", "blanc_titre"]
        result = validate_features(
            features,
            ffe_protected_attributes,
            df=sample_dataframe,
        )
        assert result.is_valid is False
        assert any("blanc_titre" in v for v in result.violations)

    def test_no_protected_in_features_all_clean(
        self,
        sample_dataframe: pd.DataFrame,
        ffe_protected_attributes: list[ProtectedAttribute],
    ) -> None:
        """Aucun attribut protege dans les features -> tout clean."""
        features = ["blanc_elo", "noir_elo", "diff_elo"]
        result = validate_features(
            features,
            ffe_protected_attributes,
            df=sample_dataframe,
        )
        assert result.is_valid is True
        assert len(result.violations) == 0
        assert len(result.warnings) == 0

    def test_validation_result_serializable(
        self,
        ffe_protected_attributes: list[ProtectedAttribute],
    ) -> None:
        """Le resultat est serialisable en JSON."""
        features = ["blanc_elo", "ligue_code"]
        result = validate_features(features, ffe_protected_attributes)
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["is_valid"] is True
