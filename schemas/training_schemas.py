"""Pandera schemas for ALICE Engine training data validation.

Based on FFE (Fédération Française des Échecs) regulations:
- R01_2025_26: Règles générales
- A02_2025_26: Championnat de France des Clubs
- J02_2025_26: Championnat de France Interclubs Jeunes
- J03_2025_26: Championnat de France scolaire
- Règlement N4 PACA 2024-2025
- Règlement Régionale PACA 2024-2025

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML (Lineage, Validation)
- ISO/IEC 42001:2023 - AI Management System
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import Any, Self

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

# =============================================================================
# FFE REGULATORY CONSTANTS (Source: règlements FFE 2025-26)
# =============================================================================

# Competition hierarchy (niveau) - A02 Art. 1.1
# NOTE: niveau is used for hierarchy (0-13) OR Elo limits for regional cups
NIVEAU_HIERARCHY = {
    "TOP16": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "N4": 4,
    "REG": 5,
    "DEP": 6,
    "TOP_JEUNES": 10,
    "N1_JEUNES": 11,
    "N2_JEUNES": 12,
    "N3_JEUNES": 13,
}

# Valid niveau for hierarchy (not Elo-based cups)
NIVEAU_HIERARCHY_MAX = 13

# Elo-based regional cups (Coupe 1500, Coupe 2000, etc.)
NIVEAU_ELO_CUPS = [1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200]

# Competition types (règlements FFE)
# NOTE: "national" removed - not present in actual data
COMPETITION_TYPES = [
    "national_feminin",
    "coupe",
    "coupe_jcl",
    "coupe_parite",
    "national_jeunes",
    "scolaire",
    "regional",
    "autre",
]

# Age categories (R01 Art. 2.4)
AGE_CATEGORIES = [
    "U8",
    "U8F",
    "U10",
    "U10F",
    "U12",
    "U12F",
    "U14",
    "U14F",
    "U16",
    "U16F",
    "U18",
    "U18F",
    "U20",
    "U20F",
    "Sen",
    "S50",
    "S65",
]

# FIDE titles (verified against actual data)
FIDE_TITLES = ["", "CM", "FM", "IM", "GM", "WFM", "WIM", "WGM"]

# Elo constraints (R01 Art. 5, FIDE Ch. 6.1)
ELO_MIN_ESTIME = 799
ELO_MAX_INITIAL = 2200
ELO_FLOOR_FIDE = 1400
ELO_MAX_REASONABLE = 2900

# Game scores by competition type
# Adultes (A02 Art. 4.1): victoire=1, nulle=0.5, défaite=0
# Jeunes éch. 1-6 (J02 Art. 4.1): victoire=2, défaite=0
# Jeunes éch. 7-8 (J02 Art. 4.1): victoire=1, défaite=0
VALID_GAME_SCORES_ADULTES = [0.0, 0.5, 1.0]
VALID_GAME_SCORES_JEUNES_HIGH = [0.0, 2.0]
VALID_GAME_SCORES_JEUNES_LOW = [0.0, 1.0]
VALID_GAME_SCORES_ALL = [0.0, 0.5, 1.0, 2.0]

# Valid result sums (resultat_blanc + resultat_noir)
# 0.0 = ajournement/forfait double
# 1.0 = Adultes standard, Jeunes éch 7-8
# 2.0 = Jeunes éch 1-6
VALID_RESULT_SUMS = [0.0, 1.0, 2.0]

# Match points (A02 Art. 4.2)
VALID_MATCH_POINTS = [0, 1, 2, 3]

# Result types (verified against actual data)
VALID_RESULT_TYPES = [
    "victoire_blanc",
    "victoire_noir",
    "nulle",
    "victoire_blanc_ajournement",
    "victoire_noir_ajournement",
    "ajournement",
]

# Strategic zones (verified against actual data)
VALID_ZONES_ENJEU = ["mi_tableau"]  # Only value in current data

# Board/round constraints
ECHIQUIER_MIN = 1
ECHIQUIER_MAX_ABSOLUTE = 16
ECHIQUIER_JEUNES_HIGH_BOARDS = 6  # J02 Art. 4.1: boards 1-6 have win=2
RONDE_MIN = 1
RONDE_MAX_ABSOLUTE = 18

# FFE Regulatory thresholds (A02 Art. 3.7.j)
NIVEAU_N4 = 4  # N4 = niveau 4
ELO_MAX_N4_PLUS = 2400  # Elo > 2400 interdit en N4+

# ISO 5259 Quality thresholds
QUALITY_VALIDATION_RATE_THRESHOLD = 0.95  # 95% valid required
MAX_SAMPLE_VALUES = 5  # Max sample values to store per error


# =============================================================================
# ISO 5259 DATA QUALITY STRUCTURES
# =============================================================================


class ErrorSeverity(Enum):
    """ISO 5259 error categorization."""

    CRITICAL = "critical"  # Data unusable
    HIGH = "high"  # Significant quality issue
    MEDIUM = "medium"  # Minor quality issue
    WARNING = "warning"  # Informational
    INFO = "info"  # Metadata


@dataclass
class DataLineage:
    """ISO 5259 data lineage tracking."""

    source_path: str
    source_hash: str
    row_count: int
    column_count: int
    validation_timestamp: str
    schema_version: str = "1.1.0"

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, source_path: str) -> Self:
        """Create lineage from DataFrame."""
        # Compute hash of first 1000 rows for performance
        sample = df.head(1000).to_json()
        data_hash = sha256(sample.encode()).hexdigest()[:16]

        return cls(
            source_path=source_path,
            source_hash=data_hash,
            row_count=len(df),
            column_count=len(df.columns),
            validation_timestamp=datetime.now().isoformat(),
        )


@dataclass
class QualityMetrics:
    """ISO 5259 data quality metrics."""

    total_rows: int
    valid_rows: int
    null_percentages: dict[str, float]
    validation_rate: float
    critical_errors: int
    high_errors: int
    medium_errors: int
    warnings: int

    @property
    def is_acceptable(self) -> bool:
        """Check if quality meets minimum threshold (95% valid)."""
        return (
            self.validation_rate >= QUALITY_VALIDATION_RATE_THRESHOLD and self.critical_errors == 0
        )


@dataclass
class ValidationError:
    """Structured validation error."""

    column: str
    check: str
    failure_count: int
    severity: ErrorSeverity
    sample_values: list[Any] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class ValidationReport:
    """ISO 5259 compliant validation report."""

    lineage: DataLineage
    metrics: QualityMetrics
    errors: list[ValidationError]
    is_valid: bool
    schema_mode: str  # "strict" or "permissive"

    def to_dict(self) -> dict:
        """Export report as dictionary for serialization."""
        return {
            "lineage": {
                "source_path": self.lineage.source_path,
                "source_hash": self.lineage.source_hash,
                "row_count": self.lineage.row_count,
                "column_count": self.lineage.column_count,
                "validation_timestamp": self.lineage.validation_timestamp,
                "schema_version": self.lineage.schema_version,
            },
            "metrics": {
                "total_rows": self.metrics.total_rows,
                "valid_rows": self.metrics.valid_rows,
                "validation_rate": self.metrics.validation_rate,
                "critical_errors": self.metrics.critical_errors,
                "high_errors": self.metrics.high_errors,
                "medium_errors": self.metrics.medium_errors,
                "warnings": self.metrics.warnings,
            },
            "errors": [
                {
                    "column": e.column,
                    "check": e.check,
                    "failure_count": e.failure_count,
                    "severity": e.severity.value,
                    "recommendation": e.recommendation,
                }
                for e in self.errors
            ],
            "is_valid": self.is_valid,
            "schema_mode": self.schema_mode,
        }


# =============================================================================
# FFE BUSINESS LOGIC CHECKS (DataFrame-level)
# =============================================================================


def _check_result_sum_valid(df: pd.DataFrame) -> pd.Series:
    """Check resultat_blanc + resultat_noir is valid.

    Valid sums:
    - 0.0: ajournement or double forfait
    - 1.0: standard game (Adultes) or Jeunes boards 7-8
    - 2.0: Jeunes boards 1-6
    """
    result_sum = df["resultat_blanc"] + df["resultat_noir"]
    return result_sum.isin(VALID_RESULT_SUMS)


def _check_blanc_equipe_coherence(df: pd.DataFrame) -> pd.Series:
    """Check blanc_equipe is either equipe_dom or equipe_ext.

    FFE: A player must belong to one of the two competing teams.
    """
    return (df["blanc_equipe"] == df["equipe_dom"]) | (df["blanc_equipe"] == df["equipe_ext"])


def _check_noir_equipe_coherence(df: pd.DataFrame) -> pd.Series:
    """Check noir_equipe is either equipe_dom or equipe_ext.

    FFE: A player must belong to one of the two competing teams.
    """
    return (df["noir_equipe"] == df["equipe_dom"]) | (df["noir_equipe"] == df["equipe_ext"])


def _check_type_resultat_coherence(df: pd.DataFrame) -> pd.Series:
    """Check type_resultat matches resultat_blanc/noir values.

    Rules:
    - victoire_blanc/victoire_blanc_ajournement: resultat_blanc > resultat_noir
    - victoire_noir/victoire_noir_ajournement: resultat_noir > resultat_blanc
    - nulle: resultat_blanc == resultat_noir (and both > 0)
    - ajournement: resultat_blanc == resultat_noir == 0
    """
    blanc_wins = df["type_resultat"].isin(["victoire_blanc", "victoire_blanc_ajournement"])
    noir_wins = df["type_resultat"].isin(["victoire_noir", "victoire_noir_ajournement"])
    is_nulle = df["type_resultat"] == "nulle"
    is_ajournement = df["type_resultat"] == "ajournement"

    # Check coherence
    blanc_wins_ok = ~blanc_wins | (df["resultat_blanc"] > df["resultat_noir"])
    noir_wins_ok = ~noir_wins | (df["resultat_noir"] > df["resultat_blanc"])
    nulle_ok = ~is_nulle | (
        (df["resultat_blanc"] == df["resultat_noir"]) & (df["resultat_blanc"] > 0)
    )
    ajournement_ok = ~is_ajournement | ((df["resultat_blanc"] == 0) & (df["resultat_noir"] == 0))

    return blanc_wins_ok & noir_wins_ok & nulle_ok & ajournement_ok


def _check_diff_elo_calculation(df: pd.DataFrame) -> pd.Series:
    """Check diff_elo equals blanc_elo - noir_elo."""
    return df["diff_elo"] == df["blanc_elo"] - df["noir_elo"]


def _check_niveau_valid(df: pd.DataFrame) -> pd.Series:
    """Check niveau is either hierarchy (0-13) or Elo-based cup value.

    WARNING: Data contains mixed usage of niveau field.
    """
    is_hierarchy = df["niveau"] <= NIVEAU_HIERARCHY_MAX
    is_elo_cup = df["niveau"].isin(NIVEAU_ELO_CUPS)
    # Accept other values with warning (historical data inconsistency)
    return is_hierarchy | is_elo_cup | (df["niveau"] >= 0)


def _check_elo_niveau_restriction(df: pd.DataFrame) -> pd.Series:
    """Check Elo restriction for lower divisions.

    A02 Art. 3.7.j: Elo > 2400 interdit en N4 et divisions inférieures.
    Only applies to hierarchy-based niveau (not Elo cups).
    """
    is_restricted = (df["niveau"] >= NIVEAU_N4) & (df["niveau"] <= NIVEAU_HIERARCHY_MAX)
    elo_ok = (df["blanc_elo"] <= ELO_MAX_N4_PLUS) & (df["noir_elo"] <= ELO_MAX_N4_PLUS)
    return ~is_restricted | elo_ok


# =============================================================================
# PANDERA SCHEMA DEFINITION
# =============================================================================


def create_training_schema(strict: bool = False) -> DataFrameSchema:
    """Create Pandera schema for training data validation.

    Args:
    ----
        strict: If True, enforce strict FFE regulatory constraints.
                If False, use permissive validation for historical data.

    Returns:
    -------
        DataFrameSchema for validation

    """
    valid_scores = VALID_GAME_SCORES_ADULTES if strict else VALID_GAME_SCORES_ALL

    # Build dataframe-level checks
    # Permissive mode: only check derived field calculation
    df_checks = [
        Check(
            _check_diff_elo_calculation,
            error="diff_elo must equal blanc_elo - noir_elo",
        ),
    ]

    # Strict mode: add all FFE business logic checks
    # NOTE: Historical data has known quality issues (type_resultat inconsistencies)
    if strict:
        df_checks.extend(
            [
                Check(
                    _check_result_sum_valid,
                    error="resultat_blanc + resultat_noir must be in [0, 1, 2]",
                ),
                Check(
                    _check_type_resultat_coherence,
                    error="type_resultat must match resultat_blanc/noir values",
                ),
                Check(
                    _check_blanc_equipe_coherence,
                    error="blanc_equipe must be equipe_dom or equipe_ext",
                ),
                Check(
                    _check_noir_equipe_coherence,
                    error="noir_equipe must be equipe_dom or equipe_ext",
                ),
                Check(
                    _check_elo_niveau_restriction,
                    error="A02 Art. 3.7.j: Elo > 2400 interdit en N4+",
                ),
            ]
        )

    return DataFrameSchema(
        columns={
            # === COMPETITION METADATA ===
            "saison": Column(
                int,
                checks=[Check.ge(2000), Check.le(2100)],
                nullable=False,
            ),
            "competition": Column(str, nullable=False),
            "division": Column(str, nullable=False),
            "groupe": Column(str, nullable=True),
            "ligue": Column(str, nullable=True),
            "ligue_code": Column(str, nullable=True),
            "niveau": Column(
                int,
                checks=[Check.ge(0)],  # Permissive: allow all >= 0
                nullable=False,
            ),
            "type_competition": Column(
                str,
                checks=Check.isin(COMPETITION_TYPES) if strict else None,
                nullable=False,
            ),
            "ronde": Column(
                int,
                checks=[Check.ge(RONDE_MIN), Check.le(RONDE_MAX_ABSOLUTE)],
                nullable=False,
            ),
            # === MATCH METADATA ===
            "equipe_dom": Column(str, nullable=False),
            "equipe_ext": Column(str, nullable=False),
            "score_dom": Column(int, checks=Check.ge(0), nullable=False),
            "score_ext": Column(int, checks=Check.ge(0), nullable=False),
            "date": Column("datetime64[us]", nullable=True),
            "date_str": Column(str, nullable=True),
            "heure": Column(str, nullable=True),
            "jour_semaine": Column(str, nullable=True),
            "lieu": Column(str, nullable=True),
            # === GAME DATA ===
            "echiquier": Column(
                int,
                checks=[Check.ge(ECHIQUIER_MIN), Check.le(ECHIQUIER_MAX_ABSOLUTE)],
                nullable=False,
            ),
            # === PLAYER DATA - WHITE ===
            "blanc_nom": Column(str, nullable=False),
            "blanc_titre": Column(
                str,
                checks=Check.isin(FIDE_TITLES) if strict else None,
                nullable=True,
            ),
            "blanc_elo": Column(
                int,
                checks=[Check.ge(ELO_MIN_ESTIME), Check.le(ELO_MAX_REASONABLE)],
                nullable=False,
            ),
            "blanc_equipe": Column(str, nullable=False),
            # === PLAYER DATA - BLACK ===
            "noir_nom": Column(str, nullable=False),
            "noir_titre": Column(
                str,
                checks=Check.isin(FIDE_TITLES) if strict else None,
                nullable=True,
            ),
            "noir_elo": Column(
                int,
                checks=[Check.ge(ELO_MIN_ESTIME), Check.le(ELO_MAX_REASONABLE)],
                nullable=False,
            ),
            "noir_equipe": Column(str, nullable=False),
            # === GAME RESULT ===
            "resultat_blanc": Column(
                float,
                checks=Check.isin(valid_scores),
                nullable=False,
            ),
            "resultat_noir": Column(
                float,
                checks=Check.isin(valid_scores),
                nullable=False,
            ),
            "resultat_text": Column(str, nullable=False),
            "type_resultat": Column(
                str,
                checks=Check.isin(VALID_RESULT_TYPES),
                nullable=False,
            ),
            # === DERIVED FEATURES ===
            "diff_elo": Column(int, nullable=False),
            # === TEAM RELIABILITY FEATURES ===
            "taux_forfait_dom": Column(
                float,
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True,
            ),
            "taux_non_joue_dom": Column(
                float,
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True,
            ),
            "fiabilite_score_dom": Column(
                float,
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True,
            ),
            "taux_forfait_ext": Column(
                float,
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True,
            ),
            "taux_non_joue_ext": Column(
                float,
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True,
            ),
            "fiabilite_score_ext": Column(
                float,
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True,
            ),
            # === PLAYER RELIABILITY FEATURES ===
            "taux_presence_blanc": Column(
                float,
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True,
            ),
            "joueur_fantome_blanc": Column(bool, nullable=True),
            "taux_presence_noir": Column(
                float,
                checks=[Check.ge(0.0), Check.le(1.0)],
                nullable=True,
            ),
            "joueur_fantome_noir": Column(bool, nullable=True),
            # === PLAYER FORM FEATURES ===
            "forme_recente_blanc": Column(float, nullable=True),
            "forme_recente_noir": Column(float, nullable=True),
            # === PLAYER POSITION FEATURES ===
            "echiquier_moyen_blanc": Column(float, nullable=True),
            "echiquier_moyen_noir": Column(float, nullable=True),
            # === FFE MULTI-TEAM FEATURES ===
            "ffe_nb_equipes_blanc": Column(int, checks=Check.ge(0), nullable=True),
            "ffe_niveau_max_blanc": Column(int, nullable=True),
            "ffe_niveau_min_blanc": Column(int, nullable=True),
            "ffe_multi_equipe_blanc": Column(bool, nullable=True),
            "ffe_nb_equipes_noir": Column(int, checks=Check.ge(0), nullable=True),
            "ffe_niveau_max_noir": Column(int, nullable=True),
            "ffe_niveau_min_noir": Column(int, nullable=True),
            "ffe_multi_equipe_noir": Column(bool, nullable=True),
            # === STRATEGIC CONTEXT FEATURES ===
            "zone_enjeu_dom": Column(str, nullable=True),
            "niveau_hier_dom": Column(int, nullable=True),
            "zone_enjeu_ext": Column(str, nullable=True),
            "niveau_hier_ext": Column(int, nullable=True),
        },
        checks=df_checks,
        coerce=True,
        strict=False,
    )


# Pre-configured schemas
TrainingSchemaPermissive = create_training_schema(strict=False)
TrainingSchemaStrict = create_training_schema(strict=True)


# =============================================================================
# VALIDATION FUNCTIONS (ISO 5259 Compliant)
# =============================================================================


def validate_training_data(
    df: pd.DataFrame,
    strict: bool = False,
    lazy: bool = True,
) -> pa.errors.SchemaErrors | None:
    """Validate training DataFrame against FFE regulatory schema.

    Args:
    ----
        df: DataFrame to validate
        strict: Use strict FFE constraints
        lazy: If True, collect all errors; if False, fail on first error

    Returns:
    -------
        None if valid, SchemaErrors if invalid (when lazy=True)

    """
    schema = TrainingSchemaStrict if strict else TrainingSchemaPermissive
    try:
        schema.validate(df, lazy=lazy)
        return None
    except pa.errors.SchemaErrors as err:
        return err


def validate_with_report(
    df: pd.DataFrame,
    source_path: str = "unknown",
    strict: bool = False,
) -> ValidationReport:
    """Validate DataFrame and return ISO 5259 compliant report.

    Args:
    ----
        df: DataFrame to validate
        source_path: Path to source file for lineage
        strict: Use strict FFE constraints

    Returns:
    -------
        ValidationReport with lineage, metrics, and errors

    """
    # Create lineage
    lineage = DataLineage.from_dataframe(df, source_path)

    # Validate
    schema = TrainingSchemaStrict if strict else TrainingSchemaPermissive
    errors: list[ValidationError] = []
    is_valid = True

    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        is_valid = False
        # Parse errors into structured format
        for _, row in err.failure_cases.iterrows():
            severity = _classify_error_severity(row["check"], row["column"])
            errors.append(
                ValidationError(
                    column=str(row["column"]) if pd.notna(row["column"]) else "schema",
                    check=str(row["check"]),
                    failure_count=1,  # Each row is one failure
                    severity=severity,
                    sample_values=[row.get("failure_case")],
                    recommendation=_get_recommendation(row["check"]),
                )
            )

    # Aggregate errors by column/check
    aggregated_errors = _aggregate_errors(errors)

    # Calculate metrics
    null_pcts = {col: df[col].isna().mean() for col in df.columns}
    critical_count = sum(1 for e in aggregated_errors if e.severity == ErrorSeverity.CRITICAL)
    high_count = sum(1 for e in aggregated_errors if e.severity == ErrorSeverity.HIGH)
    medium_count = sum(1 for e in aggregated_errors if e.severity == ErrorSeverity.MEDIUM)
    warning_count = sum(1 for e in aggregated_errors if e.severity == ErrorSeverity.WARNING)

    error_rows = sum(e.failure_count for e in aggregated_errors)
    valid_rows = max(0, len(df) - error_rows)

    metrics = QualityMetrics(
        total_rows=len(df),
        valid_rows=valid_rows,
        null_percentages=null_pcts,
        validation_rate=valid_rows / len(df) if len(df) > 0 else 1.0,
        critical_errors=critical_count,
        high_errors=high_count,
        medium_errors=medium_count,
        warnings=warning_count,
    )

    return ValidationReport(
        lineage=lineage,
        metrics=metrics,
        errors=aggregated_errors,
        is_valid=is_valid,
        schema_mode="strict" if strict else "permissive",
    )


def _classify_error_severity(check: str, column: str) -> ErrorSeverity:
    """Classify error severity based on check type."""
    check_str = str(check).lower()

    # Critical: data integrity
    if "dtype" in check_str or "coerce" in check_str:
        return ErrorSeverity.CRITICAL
    if column in ["resultat_blanc", "resultat_noir", "blanc_elo", "noir_elo"]:
        return ErrorSeverity.CRITICAL

    # High: business logic
    if "diff_elo" in check_str or "equipe" in check_str:
        return ErrorSeverity.HIGH
    if "resultat" in check_str or "type_resultat" in check_str:
        return ErrorSeverity.HIGH

    # Medium: regulatory
    if "niveau" in check_str or "competition" in check_str:
        return ErrorSeverity.MEDIUM

    return ErrorSeverity.WARNING


def _get_recommendation(check: str) -> str:
    """Get remediation recommendation for check failure."""
    check_str = str(check).lower()

    if "diff_elo" in check_str:
        return "Recalculate diff_elo as blanc_elo - noir_elo"
    if "equipe" in check_str:
        return "Verify player team assignment matches match teams"
    if "resultat" in check_str:
        return "Check game result encoding matches FFE regulations"
    if "elo" in check_str:
        return "Verify Elo rating is within valid range [799, 2900]"
    if "niveau" in check_str:
        return "Check competition level encoding"

    return "Review data against FFE regulations"


def _aggregate_errors(errors: list[ValidationError]) -> list[ValidationError]:
    """Aggregate errors by column and check."""
    aggregated: dict[tuple[str, str], ValidationError] = {}

    for error in errors:
        key = (error.column, error.check)
        if key in aggregated:
            aggregated[key].failure_count += 1
            if len(aggregated[key].sample_values) < MAX_SAMPLE_VALUES:
                aggregated[key].sample_values.extend(error.sample_values)
        else:
            aggregated[key] = ValidationError(
                column=error.column,
                check=error.check,
                failure_count=1,
                severity=error.severity,
                sample_values=error.sample_values[:5],
                recommendation=error.recommendation,
            )

    return list(aggregated.values())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_expected_score_range(type_competition: str, echiquier: int) -> list[float]:
    """Get valid score range based on competition type and board number."""
    if type_competition == "national_jeunes":
        if ECHIQUIER_MIN <= echiquier <= ECHIQUIER_JEUNES_HIGH_BOARDS:
            return VALID_GAME_SCORES_JEUNES_HIGH  # victoire=2 for boards 1-6
        return VALID_GAME_SCORES_JEUNES_LOW  # échiquiers 7-8: victoire=1
    elif type_competition == "scolaire":
        return VALID_GAME_SCORES_JEUNES_LOW
    return VALID_GAME_SCORES_ADULTES  # FIDE standard


def is_valid_niveau_for_elo(niveau: int, elo: int) -> bool:
    """Check if Elo is valid for given competition level.

    A02 Art. 3.7.j: Elo > 2400 interdit en N4 et divisions inférieures.
    """
    if niveau >= NIVEAU_N4 and niveau <= NIVEAU_HIERARCHY_MAX and elo > ELO_MAX_N4_PLUS:
        return False
    return True


def compute_quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute data quality summary for monitoring.

    Returns dict with key metrics for dashboards/logging.
    """
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "null_percentage": df.isna().mean().mean() * 100,
        "duplicate_rows": df.duplicated().sum(),
        "saison_range": [int(df["saison"].min()), int(df["saison"].max())],
        "elo_range": [
            int(min(df["blanc_elo"].min(), df["noir_elo"].min())),
            int(max(df["blanc_elo"].max(), df["noir_elo"].max())),
        ],
        "competition_types": df["type_competition"].unique().tolist(),
    }
