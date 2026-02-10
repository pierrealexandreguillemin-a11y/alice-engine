"""Validateur d'attributs proteges - ISO 24027.

Ce module valide que les features sensibles ne sont pas
utilisees directement par le modele et detecte les proxies.

Fonctions:
- validate_features: validation FORBIDDEN + PROXY_CHECK + proxies
- detect_proxy_correlations: Pearson + Cramer's V
- _cramers_v: calcul V de Cramer

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- EEOC 80% rule - Disparate impact
- ISO/IEC 5055:2021 - Code Quality (<150 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from scripts.fairness.protected.config import (
    DEFAULT_PROTECTED_ATTRIBUTES,
    PROXY_CORRELATION_THRESHOLD,
)
from scripts.fairness.protected.types import (
    ProtectedAttribute,
    ProtectionLevel,
    ProxyCorrelation,
    ValidationResult,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def validate_features(
    features: list[str],
    protected_attributes: list[ProtectedAttribute] | None = None,
    *,
    df: pd.DataFrame | None = None,
    threshold: float = PROXY_CORRELATION_THRESHOLD,
    categorical_features: list[str] | None = None,
) -> ValidationResult:
    """Valide les features contre les attributs proteges.

    Args:
    ----
        features: Liste des features utilisees par le modele
        protected_attributes: Attributs proteges a verifier
        df: DataFrame pour detection de proxies (optionnel)
        threshold: Seuil de correlation proxy
        categorical_features: Features categorielles dans df

    Returns:
    -------
        ValidationResult avec violations, warnings et proxies
    """
    if protected_attributes is None:
        protected_attributes = DEFAULT_PROTECTED_ATTRIBUTES

    violations: list[str] = []
    warnings: list[str] = []

    _check_direct_usage(features, protected_attributes, violations, warnings)

    proxy_correlations: list[ProxyCorrelation] = []
    if df is not None:
        proxy_correlations = detect_proxy_correlations(
            df,
            features,
            protected_attributes,
            threshold,
            categorical_features,
        )
        for proxy in proxy_correlations:
            warnings.append(
                f"Proxy detected: '{proxy.feature}' correlates with "
                f"'{proxy.protected_attr}' ({proxy.method}={proxy.correlation:.3f})"
            )

    is_valid = len(violations) == 0
    timestamp = datetime.now(tz=UTC).isoformat()

    result = ValidationResult(
        is_valid=is_valid,
        violations=violations,
        warnings=warnings,
        proxy_correlations=proxy_correlations,
        timestamp=timestamp,
    )
    _log_result(result)
    return result


def _check_direct_usage(
    features: list[str],
    protected_attributes: list[ProtectedAttribute],
    violations: list[str],
    warnings: list[str],
) -> None:
    """Verifie l'usage direct des attributs proteges."""
    feature_set = set(features)
    for attr in protected_attributes:
        if attr.name not in feature_set:
            continue
        if attr.level == ProtectionLevel.FORBIDDEN:
            violations.append(f"FORBIDDEN: '{attr.name}' used in features - {attr.reason}")
        elif attr.level == ProtectionLevel.PROXY_CHECK:
            warnings.append(f"PROXY_CHECK: '{attr.name}' in features - {attr.reason}")


def detect_proxy_correlations(
    df: pd.DataFrame,
    features: list[str],
    protected_attributes: list[ProtectedAttribute],
    threshold: float = PROXY_CORRELATION_THRESHOLD,
    categorical_features: list[str] | None = None,
) -> list[ProxyCorrelation]:
    """Detecte les correlations proxy entre features et attributs proteges.

    Utilise Pearson pour numerique, Cramer's V pour categoriel.
    """
    categorical_features = categorical_features or []
    correlations: list[ProxyCorrelation] = []

    for attr in protected_attributes:
        if attr.name not in df.columns:
            continue
        for feature in features:
            if feature not in df.columns or feature == attr.name:
                continue
            corr, method = _compute_correlation(
                df,
                feature,
                attr.name,
                feature in categorical_features,
            )
            if abs(corr) >= threshold:
                correlations.append(
                    ProxyCorrelation(
                        feature=feature,
                        protected_attr=attr.name,
                        correlation=round(abs(corr), 4),
                        method=method,
                    )
                )
    return correlations


def _compute_correlation(
    df: pd.DataFrame,
    feature: str,
    protected: str,
    is_categorical: bool,
) -> tuple[float, str]:
    """Calcule la correlation entre une feature et un attribut protege.

    Note: pour numeric-vs-categorical, utilise Pearson apres LabelEncoding.
    Valide pour ordinal; pour nominal pur, Cramer's V serait preferable.
    """
    if is_categorical or df[feature].dtype == object:
        return _cramers_v(df[feature].values, df[protected].values), "cramers_v"
    if df[protected].dtype == object:
        from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415

        encoded = LabelEncoder().fit_transform(df[protected].astype(str))
        corr = np.corrcoef(df[feature].values.astype(float), encoded)[0, 1]
        return (0.0 if np.isnan(corr) else float(corr)), "pearson"
    corr = np.corrcoef(
        df[feature].values.astype(float),
        df[protected].values.astype(float),
    )[0, 1]
    return (0.0 if np.isnan(corr) else float(corr)), "pearson"


def _cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """Calcule le V de Cramer standard entre deux variables categorielles.

    Retourne 0 si une colonne n'a qu'une seule valeur.
    Note: pas de correction de biais (Bergsma-Wicher). Pour de grands
    echantillons (n>500) le biais est negligeable.
    """
    import pandas as pd  # noqa: PLC0415
    from scipy.stats import chi2_contingency  # noqa: PLC0415

    confusion = pd.crosstab(x, y)
    if confusion.shape[0] <= 1 or confusion.shape[1] <= 1:
        return 0.0

    chi2, _, _, _ = chi2_contingency(confusion)
    n = len(x)
    min_dim = min(confusion.shape[0], confusion.shape[1]) - 1
    if min_dim == 0 or n == 0:
        return 0.0

    return float(np.sqrt(chi2 / (n * min_dim)))


def _log_result(result: ValidationResult) -> None:
    """Log le resultat de la validation."""
    if result.violations:
        logger.warning("Protected attrs: %d violations", len(result.violations))
        for v in result.violations:
            logger.warning("  %s", v)
    if result.warnings:
        logger.info("Protected attrs: %d warnings", len(result.warnings))
        for w in result.warnings:
            logger.info("  %s", w)
    if not result.violations and not result.warnings:
        logger.info("Protected attrs: validation clean")
