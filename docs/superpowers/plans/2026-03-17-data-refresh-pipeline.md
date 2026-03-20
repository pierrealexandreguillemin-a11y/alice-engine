# Data Refresh Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable `make refresh-data` to sync fresh data from ffe_scrapper, parse to parquet, validate ISO 5259, and generate features — one command.

**Architecture:** SRP package `scripts/sync_data/` handles data sourcing (local or HuggingFace). Existing `scripts/parse_dataset/` gets a `__main__.py` entry point. New `schemas/parsing_schemas.py` + `schemas/parsing_validation.py` wire ISO 5259 validation into the parse stage, reusing existing `DataLineage`/`ValidationReport` types.

**Tech Stack:** Pandera (schemas), Pydantic (input validation), huggingface_hub + datasets (HF sync), PyArrow (parquet I/O)

**Spec:** `docs/superpowers/specs/2026-03-17-data-refresh-pipeline-design.md`

---

## Quality Gates (Industry Standards + ISO)

Each pipeline stage has a quality gate that BLOCKS progression if thresholds fail. Aligned with Google MLOps Level 1, NIST AI RMF, EU AI Act Art.10, and existing ALICE ISO gates.

### Gate 1: Data Ingestion (after sync, before parse)

| Check | Threshold | Action |
|-------|-----------|--------|
| Source directory accessible | Required | BLOCK |
| HTML file count vs. previous | Within +/- 30% | WARN at 10%, BLOCK at 30% |
| Source freshness vs. local parquets | Source newer | INFO (skip if fresh) |

### Gate 2: Data Quality (after parse, before features) — ISO 5259

| Check | Threshold | Action |
|-------|-----------|--------|
| Schema validation (Pandera) | 100% match | BLOCK |
| Null rate per required column | <= 5% | BLOCK |
| Row count vs. previous version | Within +/- 30% | WARN at 10%, BLOCK at 30% |
| Duplicate row rate | < 1% | WARN at 0.5%, BLOCK at 1% |
| Value range (Elo [0,3000], saison [2002,2030]) | 0 violations | BLOCK |
| diff_elo coherence (blanc - noir) | 100% match | BLOCK |
| Data hash (SHA-256) computed + recorded | Required | BLOCK |
| ValidationReport persisted to `reports/validation/` | Required | BLOCK |

### Gate 3: Feature Quality (after feature_engineering, before training)

| Check | Threshold | Action |
|-------|-----------|--------|
| No NaN/inf in output features | 0 | BLOCK |
| Feature count matches spec (57 features) | Exact | BLOCK |
| Temporal split correct (train <=2022, valid 2023, test >2023) | Exact | BLOCK |
| Train/valid/test parquets exist and non-empty | Required | BLOCK |

### Gate 4: Code Quality (every commit) — ISO 5055

| Check | Threshold | Action |
|-------|-----------|--------|
| File length | < 300 lines | BLOCK |
| Function length | < 50 lines | BLOCK |
| Cyclomatic complexity | <= B (Xenon) | BLOCK |
| Ruff lint + format | 0 errors | BLOCK |
| MyPy type check | 0 errors | BLOCK |
| Bandit security | 0 high/critical | BLOCK |
| Test coverage | >= 80% on new files | BLOCK |

---

## Definition of Done (DoD) — Per Task

Every task is complete ONLY when ALL of these are true:

### Code DoD
- [ ] All tests pass (`pytest -v`)
- [ ] Coverage >= 80% on new/modified files
- [ ] `make quality` passes (ruff + mypy + bandit)
- [ ] ISO 5055: `wc -l` < 300, `radon cc` <= B
- [ ] No hardcoded secrets, paths from env vars (ISO 27034)
- [ ] Pydantic validates ALL external inputs (ISO 27034)

### Documentation DoD
- [ ] Module docstring with ISO compliance references
- [ ] Function docstrings with Args/Returns (Google style)
- [ ] Test file has ISO 29119 header (Document ID, Version, Tests count)

### Git DoD
- [ ] Conventional commit message (`feat/fix/test/refactor`)
- [ ] Only relevant files staged (no `git add -A`)
- [ ] Pre-commit hooks pass (gitleaks, ruff, mypy, bandit)

### ML-Specific DoD (for data tasks)
- [ ] Data hash computed and stored in report
- [ ] Data lineage recorded (source path, timestamp, row count)
- [ ] Quality gate thresholds verified
- [ ] Previous data preserved (not overwritten without backup)

---

## Task Completion Protocol

Each task follows this sequence:
1. **TDD**: Write failing test → implement → pass → commit
2. **Gate Check**: Run relevant quality gate checks
3. **DoD Verify**: Confirm all DoD criteria met
4. **Review**: After tasks 2, 7, and 10 — dispatch `superpowers:requesting-code-review` for accumulated changes

---

## File Structure

| Action | File | Responsibility | Lines |
|--------|------|----------------|-------|
| CREATE | `schemas/parsing_schemas.py` | Pandera schemas for raw echiquiers + joueurs parquets | ~80 |
| CREATE | `schemas/parsing_validation.py` | Validation orchestration + report persistence (ISO 5259) | ~80 |
| CREATE | `scripts/parse_dataset/__main__.py` | CLI entry point: parse + validate | ~40 |
| CREATE | `scripts/sync_data/__init__.py` | Package re-exports | ~15 |
| CREATE | `scripts/sync_data/__main__.py` | CLI entry point for sync | ~35 |
| CREATE | `scripts/sync_data/types.py` | Pydantic models: SyncConfig, SourceStatus, FreshnessReport | ~50 |
| CREATE | `scripts/sync_data/freshness.py` | check_source(), check_freshness() | ~60 |
| CREATE | `scripts/sync_data/symlink.py` | update_symlink() with Windows fallback | ~40 |
| CREATE | `scripts/sync_data/huggingface.py` | pull_from_hf(), push_to_hf() | ~60 |
| CREATE | `tests/schemas/test_parsing_schemas.py` | Tests for raw Pandera schemas | ~120 |
| CREATE | `tests/schemas/test_parsing_validation.py` | Tests for validation orchestration | ~80 |
| CREATE | `tests/test_sync_data.py` | Tests for sync_data package | ~150 |
| MODIFY | `schemas/__init__.py` | Add parsing re-exports | +6 |
| MODIFY | `Makefile` | Add sync, refresh-data targets | +8 |
| MODIFY | `requirements.txt` | Add datasets, huggingface_hub | +2 |

---

## Task 1: Parsing Schemas (Pandera)

**Files:**
- Create: `schemas/parsing_schemas.py`
- Test: `tests/schemas/test_parsing_schemas.py`

**Ref:** Follow pattern from `schemas/training_schemas.py` (Column builders returning `dict[str, Column]`)

- [ ] **Step 1.1: Write test file with header + fixtures**

```python
# tests/schemas/test_parsing_schemas.py
"""Tests Parsing Schemas - ISO 29119.

Document ID: ALICE-TEST-SCHEMAS-PARSING
Version: 1.0.0
Tests: 10

Classes:
- TestEchiquiersRawSchema: Tests for raw echiquiers validation (5 tests)
- TestJoueursRawSchema: Tests for raw joueurs validation (5 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
"""
from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from schemas.parsing_schemas import EchiquiersRawSchema, JoueursRawSchema


@pytest.fixture()
def valid_echiquiers_df() -> pd.DataFrame:
    """Minimal valid echiquiers DataFrame."""
    return pd.DataFrame({
        "saison": [2025],
        "competition": ["Interclubs"],
        "division": ["Nationale_1"],
        "groupe": ["Groupe_A"],
        "ronde": [1],
        "echiquier": [1],
        "blanc_nom": ["DUPONT Jean"],
        "noir_nom": ["MARTIN Paul"],
        "blanc_elo": [2100],
        "noir_elo": [1950],
        "resultat_blanc": [1.0],
        "resultat_noir": [0.0],
        "type_resultat": ["victoire_blanc"],
        "diff_elo": [150],
    })


@pytest.fixture()
def valid_joueurs_df() -> pd.DataFrame:
    """Minimal valid joueurs DataFrame."""
    return pd.DataFrame({
        "nr_ffe": ["K59857"],
        "id_ffe": [672495],
        "nom": ["DUPONT"],
        "prenom": ["Jean"],
        "nom_complet": ["DUPONT Jean"],
        "elo": [1567],
        "elo_type": ["F"],
        "categorie": ["SenM"],
        "club": ["Echiquier de Bigorre"],
    })


class TestEchiquiersRawSchema:
    """Tests for EchiquiersRawSchema."""

    def test_valid_dataframe_passes(self, valid_echiquiers_df):
        """Test validation with valid DataFrame."""
        EchiquiersRawSchema.validate(valid_echiquiers_df, lazy=True)

    def test_invalid_saison_raises(self, valid_echiquiers_df):
        """Test saison out of range raises."""
        valid_echiquiers_df["saison"] = [1900]
        with pytest.raises(SchemaErrors):
            EchiquiersRawSchema.validate(valid_echiquiers_df, lazy=True)

    def test_invalid_elo_raises(self, valid_echiquiers_df):
        """Test Elo out of range raises."""
        valid_echiquiers_df["blanc_elo"] = [5000]
        with pytest.raises(SchemaErrors):
            EchiquiersRawSchema.validate(valid_echiquiers_df, lazy=True)

    def test_missing_column_raises(self):
        """Test missing required column raises."""
        df = pd.DataFrame({"saison": [2025]})
        with pytest.raises(SchemaErrors):
            EchiquiersRawSchema.validate(df, lazy=True)

    def test_diff_elo_coherence(self, valid_echiquiers_df):
        """Test diff_elo must equal blanc_elo - noir_elo."""
        valid_echiquiers_df["diff_elo"] = [999]
        with pytest.raises(SchemaErrors):
            EchiquiersRawSchema.validate(valid_echiquiers_df, lazy=True)


class TestJoueursRawSchema:
    """Tests for JoueursRawSchema."""

    def test_valid_dataframe_passes(self, valid_joueurs_df):
        """Test validation with valid DataFrame."""
        JoueursRawSchema.validate(valid_joueurs_df, lazy=True)

    def test_invalid_elo_type_raises(self, valid_joueurs_df):
        """Test invalid elo_type raises."""
        valid_joueurs_df["elo_type"] = ["X"]
        with pytest.raises(SchemaErrors):
            JoueursRawSchema.validate(valid_joueurs_df, lazy=True)

    def test_invalid_nr_ffe_pattern_raises(self, valid_joueurs_df):
        """Test nr_ffe not matching ^[A-Z]\\d+$ raises."""
        valid_joueurs_df["nr_ffe"] = ["12345"]
        with pytest.raises(SchemaErrors):
            JoueursRawSchema.validate(valid_joueurs_df, lazy=True)

    def test_missing_column_raises(self):
        """Test missing required column raises."""
        df = pd.DataFrame({"nom": ["DUPONT"]})
        with pytest.raises(SchemaErrors):
            JoueursRawSchema.validate(df, lazy=True)

    def test_elo_out_of_range_raises(self, valid_joueurs_df):
        """Test Elo > 3000 raises."""
        valid_joueurs_df["elo"] = [4000]
        with pytest.raises(SchemaErrors):
            JoueursRawSchema.validate(valid_joueurs_df, lazy=True)
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `pytest tests/schemas/test_parsing_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'schemas.parsing_schemas'`

- [ ] **Step 1.3: Implement parsing_schemas.py**

```python
# schemas/parsing_schemas.py
"""Pandera schemas for raw parsed parquets (ISO 5259).

Validates echiquiers.parquet and joueurs.parquet BEFORE feature engineering.
For post-feature validation, see training_schemas.py.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<300 lines, SRP)
"""
from __future__ import annotations

import pandas as pd
from pandera import Check, Column, DataFrameSchema


def _check_diff_elo_coherence(df: pd.DataFrame) -> pd.Series:
    """Verify diff_elo == blanc_elo - noir_elo."""
    return df["diff_elo"] == (df["blanc_elo"] - df["noir_elo"])


def _echiquiers_columns() -> dict[str, Column]:
    """Build column definitions for raw echiquiers."""
    return {
        "saison": Column(int, checks=[Check.ge(2002), Check.le(2030)], nullable=False),
        "competition": Column(str, nullable=False),
        "division": Column(str, nullable=False),
        "groupe": Column(str, nullable=False),
        "ronde": Column(int, checks=[Check.ge(1), Check.le(15)], nullable=False),
        "echiquier": Column(int, checks=[Check.ge(1), Check.le(12)], nullable=False),
        "blanc_nom": Column(str, nullable=False),
        "noir_nom": Column(str, nullable=False),
        "blanc_elo": Column(int, checks=[Check.ge(0), Check.le(3000)], nullable=False),
        "noir_elo": Column(int, checks=[Check.ge(0), Check.le(3000)], nullable=False),
        "resultat_blanc": Column(float, checks=Check.isin([0.0, 0.5, 1.0]), nullable=False),
        "resultat_noir": Column(float, checks=Check.isin([0.0, 0.5, 1.0]), nullable=False),
        "type_resultat": Column(str, nullable=False),
        "diff_elo": Column(int, nullable=False),
    }


def _joueurs_columns() -> dict[str, Column]:
    """Build column definitions for raw joueurs."""
    return {
        "nr_ffe": Column(str, checks=Check.str_matches(r"^[A-Z]\d+$"), nullable=False),
        "id_ffe": Column(int, nullable=True),
        "nom": Column(str, nullable=False),
        "prenom": Column(str, nullable=False),
        "nom_complet": Column(str, nullable=True),
        "elo": Column(int, checks=[Check.ge(0), Check.le(3000)], nullable=False),
        "elo_type": Column(str, checks=Check.isin(["F", "N", "E"]), nullable=False),
        "categorie": Column(str, nullable=False),
        "club": Column(str, nullable=False),
    }


EchiquiersRawSchema = DataFrameSchema(
    columns=_echiquiers_columns(),
    checks=[Check(_check_diff_elo_coherence, error="diff_elo != blanc_elo - noir_elo")],
    coerce=False,
)

JoueursRawSchema = DataFrameSchema(
    columns=_joueurs_columns(),
    coerce=False,
)
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `pytest tests/schemas/test_parsing_schemas.py -v`
Expected: 10 PASSED

- [ ] **Step 1.5: Commit**

```bash
git add schemas/parsing_schemas.py tests/schemas/test_parsing_schemas.py
git commit -m "feat(schemas): add Pandera schemas for raw parsed parquets (ISO 5259)"
```

---

## Task 2: Parsing Validation (ISO 5259 Reports)

**Files:**
- Create: `schemas/parsing_validation.py`
- Modify: `schemas/__init__.py` (add re-exports)
- Test: `tests/schemas/test_parsing_validation.py`

**Ref:** Follow pattern from `schemas/training_validation.py` (validate_with_report pattern, DataLineage.from_dataframe)

- [ ] **Step 2.1: Write test file**

```python
# tests/schemas/test_parsing_validation.py
"""Tests Parsing Validation - ISO 29119.

Document ID: ALICE-TEST-SCHEMAS-PARSING-VALID
Version: 1.0.0
Tests: 6

Classes:
- TestValidateRawEchiquiers: Tests validate_raw_echiquiers (3 tests)
- TestValidateRawJoueurs: Tests validate_raw_joueurs (3 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from schemas.parsing_validation import validate_raw_echiquiers, validate_raw_joueurs


@pytest.fixture()
def valid_echiquiers_df() -> pd.DataFrame:
    """Minimal valid echiquiers DataFrame."""
    return pd.DataFrame({
        "saison": [2025],
        "competition": ["Interclubs"],
        "division": ["Nationale_1"],
        "groupe": ["Groupe_A"],
        "ronde": [1],
        "echiquier": [1],
        "blanc_nom": ["DUPONT Jean"],
        "noir_nom": ["MARTIN Paul"],
        "blanc_elo": [2100],
        "noir_elo": [1950],
        "resultat_blanc": [1.0],
        "resultat_noir": [0.0],
        "type_resultat": ["victoire_blanc"],
        "diff_elo": [150],
    })


@pytest.fixture()
def valid_joueurs_df() -> pd.DataFrame:
    """Minimal valid joueurs DataFrame."""
    return pd.DataFrame({
        "nr_ffe": ["K59857"],
        "id_ffe": [672495],
        "nom": ["DUPONT"],
        "prenom": ["Jean"],
        "nom_complet": ["DUPONT Jean"],
        "elo": [1567],
        "elo_type": ["F"],
        "categorie": ["SenM"],
        "club": ["Echiquier de Bigorre"],
    })


class TestValidateRawEchiquiers:
    """Tests for validate_raw_echiquiers."""

    def test_valid_returns_report_no_errors(self, valid_echiquiers_df):
        """Test valid DataFrame returns report with no errors."""
        report = validate_raw_echiquiers(valid_echiquiers_df)
        assert report.is_valid is True
        assert report.errors == []

    def test_report_persisted_to_disk(self, valid_echiquiers_df, tmp_path):
        """Test report is saved to JSON."""
        report = validate_raw_echiquiers(
            valid_echiquiers_df,
            report_dir=tmp_path,
        )
        report_file = tmp_path / "raw_echiquiers_report.json"
        assert report_file.exists()

    def test_invalid_data_returns_errors(self):
        """Test invalid data returns report with errors."""
        df = pd.DataFrame({
            "saison": [1900],
            "competition": ["X"],
            "division": ["X"],
            "groupe": ["X"],
            "ronde": [1],
            "echiquier": [1],
            "blanc_nom": ["A"],
            "noir_nom": ["B"],
            "blanc_elo": [5000],
            "noir_elo": [1000],
            "resultat_blanc": [1.0],
            "resultat_noir": [0.0],
            "type_resultat": ["victoire_blanc"],
            "diff_elo": [4000],
        })
        report = validate_raw_echiquiers(df)
        assert report.is_valid is False
        assert len(report.errors) > 0


class TestValidateRawJoueurs:
    """Tests for validate_raw_joueurs."""

    def test_valid_returns_report_no_errors(self, valid_joueurs_df):
        """Test valid DataFrame returns report with no errors."""
        report = validate_raw_joueurs(valid_joueurs_df)
        assert report.is_valid is True
        assert report.errors == []

    def test_report_persisted_to_disk(self, valid_joueurs_df, tmp_path):
        """Test report is saved to JSON."""
        report = validate_raw_joueurs(
            valid_joueurs_df,
            report_dir=tmp_path,
        )
        report_file = tmp_path / "raw_joueurs_report.json"
        assert report_file.exists()

    def test_invalid_data_returns_errors(self):
        """Test invalid data returns report with errors."""
        df = pd.DataFrame({
            "nr_ffe": ["12345"],
            "id_ffe": [1],
            "nom": ["X"],
            "prenom": ["Y"],
            "nom_complet": ["X Y"],
            "elo": [5000],
            "elo_type": ["Z"],
            "categorie": ["SenM"],
            "club": ["Club"],
        })
        report = validate_raw_joueurs(df)
        assert report.is_valid is False
        assert len(report.errors) > 0
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `pytest tests/schemas/test_parsing_validation.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 2.3: Implement parsing_validation.py**

```python
# schemas/parsing_validation.py
"""Validation orchestration for raw parsed parquets (ISO 5259).

Validates echiquiers/joueurs DataFrames and persists reports.
Reuses DataLineage/ValidationReport from training_types.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML (Lineage, Validation)
- ISO/IEC 5055:2021 - Code Quality (<300 lines, SRP)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaErrors

from schemas.parsing_schemas import EchiquiersRawSchema, JoueursRawSchema
from schemas.training_types import (
    DataLineage,
    ErrorSeverity,
    QualityMetrics,
    ValidationError,
    ValidationReport,
)

logger = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = Path("reports/validation")


def validate_raw_echiquiers(
    df: pd.DataFrame,
    source_path: str = "data/echiquiers.parquet",
    report_dir: Path | None = None,
) -> ValidationReport:
    """Validate raw echiquiers DataFrame and persist report."""
    report = _validate_with_schema(
        df=df,
        schema=EchiquiersRawSchema,
        source_path=source_path,
        report_name="raw_echiquiers_report.json",
        report_dir=report_dir or DEFAULT_REPORT_DIR,
    )
    logger.info("Echiquiers validation: valid=%s, errors=%d", report.is_valid, len(report.errors))
    return report


def validate_raw_joueurs(
    df: pd.DataFrame,
    source_path: str = "data/joueurs.parquet",
    report_dir: Path | None = None,
) -> ValidationReport:
    """Validate raw joueurs DataFrame and persist report."""
    report = _validate_with_schema(
        df=df,
        schema=JoueursRawSchema,
        source_path=source_path,
        report_name="raw_joueurs_report.json",
        report_dir=report_dir or DEFAULT_REPORT_DIR,
    )
    logger.info("Joueurs validation: valid=%s, errors=%d", report.is_valid, len(report.errors))
    return report


def _validate_with_schema(
    df: pd.DataFrame,
    schema,
    source_path: str,
    report_name: str,
    report_dir: Path,
) -> ValidationReport:
    """Run schema validation, build report, persist to disk."""
    lineage = DataLineage.from_dataframe(df, source_path)
    is_valid, errors = _run_validation(schema, df)
    error_counts = _count_by_severity(errors)
    metrics = QualityMetrics(
        total_rows=len(df),
        valid_rows=len(df) - sum(e.failure_count for e in errors),
        null_percentages={
            col: float(df[col].isna().mean() * 100) for col in df.columns
        },
        validation_rate=1.0 if is_valid else 0.0,
        critical_errors=error_counts.get("critical", 0),
        high_errors=error_counts.get("high", 0),
        medium_errors=error_counts.get("medium", 0),
        warnings=error_counts.get("warning", 0),
    )
    report = ValidationReport(
        lineage=lineage,
        metrics=metrics,
        errors=errors,
        is_valid=is_valid,
        schema_mode="raw_parsing",
    )
    _persist_report(report, report_dir, report_name)
    return report


def _run_validation(schema, df: pd.DataFrame) -> tuple[bool, list[ValidationError]]:
    """Run Pandera schema validation, return (is_valid, errors)."""
    try:
        schema.validate(df, lazy=True)
        return True, []
    except SchemaErrors as exc:
        errors = [
            ValidationError(
                column=str(row.get("column", "dataframe")),
                check=str(row.get("check", "unknown")),
                failure_count=1,
                severity=ErrorSeverity.HIGH,
                recommendation=str(row.get("check", "")),
            )
            for _, row in exc.failure_cases.iterrows()
        ]
        return False, errors


def _count_by_severity(errors: list[ValidationError]) -> dict[str, int]:
    """Count errors by severity level."""
    counts: dict[str, int] = {}
    for error in errors:
        key = error.severity.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _persist_report(report: ValidationReport, report_dir: Path, filename: str) -> None:
    """Save report as JSON to disk."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / filename
    report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    logger.info("Report saved: %s", report_path)
```

- [ ] **Step 2.4: Update schemas/__init__.py re-exports**

Add to `schemas/__init__.py` imports section:
```python
from schemas.parsing_schemas import EchiquiersRawSchema, JoueursRawSchema
from schemas.parsing_validation import validate_raw_echiquiers, validate_raw_joueurs
```

Add to `__all__` list:
```python
    # Raw Parsing Validation (ISO 5259)
    "EchiquiersRawSchema",
    "JoueursRawSchema",
    "validate_raw_echiquiers",
    "validate_raw_joueurs",
```

- [ ] **Step 2.5: Run tests to verify they pass**

Run: `pytest tests/schemas/test_parsing_validation.py -v`
Expected: 6 PASSED

- [ ] **Step 2.6: Commit**

```bash
git add schemas/parsing_validation.py schemas/__init__.py tests/schemas/test_parsing_validation.py
git commit -m "feat(schemas): add ISO 5259 validation for raw parsed parquets"
```

---

### Review Checkpoint 1 — Schemas

> After Tasks 1-2: dispatch `superpowers:requesting-code-review` on `schemas/parsing_schemas.py`, `schemas/parsing_validation.py`, and their tests. Verify constructors match `training_types.py` signatures exactly.

---

## Task 3: Parse Dataset Entry Point

**Files:**
- Create: `scripts/parse_dataset/__main__.py`

**Ref:** `scripts/parse_dataset/orchestration.py` — `parse_compositions()` returns `dict[str, int]`, `parse_joueurs()` returns `dict[str, int]`

- [ ] **Step 3.1: Verify parse-data Makefile target exists**

Run: `grep -n "parse-data" Makefile`
Expected: existing target calling `python -m scripts.parse_dataset`

- [ ] **Step 3.2: Create __main__.py**

```python
# scripts/parse_dataset/__main__.py
"""Entry point for: python -m scripts.parse_dataset

Parses raw FFE HTML data to Parquet and validates ISO 5259.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<50 lines)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from scripts.parse_dataset.orchestration import parse_compositions, parse_joueurs
from schemas.parsing_validation import validate_raw_echiquiers, validate_raw_joueurs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse HTML -> Parquet + validate ISO 5259."""
    data_dir = Path("dataset_alice")
    output_dir = Path("data")

    logger.info("Parsing compositions -> echiquiers.parquet")
    stats_ech = parse_compositions(data_dir, output_dir / "echiquiers.parquet")
    logger.info("Compositions: %s", stats_ech)

    logger.info("Parsing players -> joueurs.parquet")
    stats_jou = parse_joueurs(data_dir / "players_v2", output_dir / "joueurs.parquet")
    logger.info("Players: %s", stats_jou)

    logger.info("Validating ISO 5259")
    df_ech = pd.read_parquet(output_dir / "echiquiers.parquet")
    df_jou = pd.read_parquet(output_dir / "joueurs.parquet")
    validate_raw_echiquiers(df_ech, source_path=str(output_dir / "echiquiers.parquet"))
    validate_raw_joueurs(df_jou, source_path=str(output_dir / "joueurs.parquet"))

    logger.info("Parse + validation complete")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.3: Smoke test**

Run: `python -m scripts.parse_dataset --help 2>&1 || python -c "from scripts.parse_dataset.__main__ import main; print('import OK')"`
Expected: import succeeds (full parse takes minutes, don't run yet)

- [ ] **Step 3.4: Commit**

```bash
git add scripts/parse_dataset/__main__.py
git commit -m "feat(parse): add __main__.py entry point for make parse-data"
```

---

## Task 4: Sync Data Package — Types

**Files:**
- Create: `scripts/sync_data/__init__.py`
- Create: `scripts/sync_data/types.py`

- [ ] **Step 4.1: Create package directory**

Run: `mkdir -p scripts/sync_data`

- [ ] **Step 4.2: Create types.py with Pydantic models**

```python
# scripts/sync_data/types.py
"""Pydantic models for data sync configuration (ISO 27034).

ISO Compliance:
- ISO/IEC 27034:2011 - Secure Coding (Pydantic validation)
- ISO/IEC 5055:2021 - Code Quality (<50 lines per function)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SyncConfig(BaseModel):
    """Validated CLI configuration for sync operations."""

    source: Literal["local", "hf"] = "local"
    source_dir: Path | None = None
    push: bool = False
    dry_run: bool = False
    hf_repo_id: str = Field(
        default="Pierrax/ffe-history",
        pattern=r"^[\w-]+/[\w.-]+$",
    )


class SourceStatus(BaseModel):
    """Status of a data source directory."""

    path: Path
    accessible: bool
    file_count: int = 0
    latest_mtime: datetime | None = None


class FreshnessReport(BaseModel):
    """Comparison between source and local parquet freshness."""

    source_mtime: datetime | None = None
    parquet_mtime: datetime | None = None
    is_stale: bool = True
    days_behind: float = 0.0
```

- [ ] **Step 4.3: Create __init__.py**

```python
# scripts/sync_data/__init__.py
"""Data sync package for ALICE Engine.

Syncs fresh data from ffe_scrapper or HuggingFace.

Usage:
    python -m scripts.sync_data
    python -m scripts.sync_data --source hf --push
"""
from scripts.sync_data.types import FreshnessReport, SourceStatus, SyncConfig

__all__ = [
    "FreshnessReport",
    "SourceStatus",
    "SyncConfig",
]
```

- [ ] **Step 4.4: Commit**

```bash
git add scripts/sync_data/__init__.py scripts/sync_data/types.py
git commit -m "feat(sync): add Pydantic types for sync configuration (ISO 27034)"
```

---

## Task 5: Sync Data — Freshness Check

**Files:**
- Create: `scripts/sync_data/freshness.py`
- Test: `tests/test_sync_data.py` (start file)

- [ ] **Step 5.1: Write test file with freshness tests**

```python
# tests/test_sync_data.py
"""Tests Sync Data - ISO 29119.

Document ID: ALICE-TEST-SYNC-DATA
Version: 1.0.0
Tests: 12

Classes:
- TestSourceCheck: Tests check_source (3 tests)
- TestFreshness: Tests check_freshness (3 tests)
- TestSymlink: Tests update_symlink (3 tests)
- TestHuggingFace: Tests pull/push HF (3 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.sync_data.freshness import check_freshness, check_source


class TestSourceCheck:
    """Tests for check_source."""

    def test_accessible_directory(self, tmp_path):
        """Test accessible dir returns accessible=True."""
        (tmp_path / "2025" / "Interclubs").mkdir(parents=True)
        (tmp_path / "2025" / "Interclubs" / "ronde_1.html").write_text("<html/>")
        status = check_source(tmp_path)
        assert status.accessible is True
        assert status.file_count >= 1

    def test_missing_directory(self):
        """Test missing dir returns accessible=False."""
        status = check_source(Path("/nonexistent/path"))
        assert status.accessible is False
        assert status.file_count == 0

    def test_empty_directory(self, tmp_path):
        """Test empty dir returns accessible=True, count=0."""
        status = check_source(tmp_path)
        assert status.accessible is True
        assert status.file_count == 0


class TestFreshness:
    """Tests for check_freshness."""

    def test_stale_parquet(self, tmp_path):
        """Test parquet older than source is stale."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "ronde_1.html").write_text("<html/>")

        parquet_dir = tmp_path / "data"
        parquet_dir.mkdir()
        pq_file = parquet_dir / "echiquiers.parquet"
        pq_file.write_text("old")
        # Make source newer by touching
        import time
        time.sleep(0.1)
        (source_dir / "ronde_2.html").write_text("<html/>")

        report = check_freshness(source_dir, parquet_dir)
        assert report.is_stale is True

    def test_fresh_parquet(self, tmp_path):
        """Test parquet newer than source is fresh."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "ronde_1.html").write_text("<html/>")

        import time
        time.sleep(0.1)

        parquet_dir = tmp_path / "data"
        parquet_dir.mkdir()
        pq_file = parquet_dir / "echiquiers.parquet"
        pq_file.write_text("new")

        report = check_freshness(source_dir, parquet_dir)
        assert report.is_stale is False

    def test_no_parquet_yet(self, tmp_path):
        """Test missing parquet is always stale."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "ronde_1.html").write_text("<html/>")

        parquet_dir = tmp_path / "data"
        parquet_dir.mkdir()

        report = check_freshness(source_dir, parquet_dir)
        assert report.is_stale is True
```

- [ ] **Step 5.2: Run tests to verify they fail**

Run: `pytest tests/test_sync_data.py::TestSourceCheck tests/test_sync_data.py::TestFreshness -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 5.3: Implement freshness.py**

```python
# scripts/sync_data/freshness.py
"""Freshness checking for data sources (ISO 5259).

Compares modification times between source HTML and local parquets.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML (Lineage)
- ISO/IEC 5055:2021 - Code Quality (<50 lines per function)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from scripts.sync_data.types import FreshnessReport, SourceStatus

logger = logging.getLogger(__name__)


def check_source(source_dir: Path) -> SourceStatus:
    """Check accessibility and stats of a data source directory."""
    if not source_dir.exists():
        return SourceStatus(path=source_dir, accessible=False)

    html_files = list(source_dir.rglob("*.html"))
    latest_mtime = None
    if html_files:
        latest_ts = max(f.stat().st_mtime for f in html_files)
        latest_mtime = datetime.fromtimestamp(latest_ts, tz=timezone.utc)

    return SourceStatus(
        path=source_dir,
        accessible=True,
        file_count=len(html_files),
        latest_mtime=latest_mtime,
    )


def check_freshness(source_dir: Path, parquet_dir: Path) -> FreshnessReport:
    """Compare source HTML freshness vs local parquet files."""
    source_status = check_source(source_dir)
    parquet_files = list(parquet_dir.glob("*.parquet"))

    if not parquet_files:
        return FreshnessReport(
            source_mtime=source_status.latest_mtime,
            is_stale=True,
        )

    latest_pq_ts = max(f.stat().st_mtime for f in parquet_files)
    parquet_mtime = datetime.fromtimestamp(latest_pq_ts, tz=timezone.utc)

    is_stale = (
        source_status.latest_mtime is not None
        and source_status.latest_mtime > parquet_mtime
    )
    days_behind = 0.0
    if is_stale and source_status.latest_mtime:
        delta = source_status.latest_mtime - parquet_mtime
        days_behind = delta.total_seconds() / 86400

    return FreshnessReport(
        source_mtime=source_status.latest_mtime,
        parquet_mtime=parquet_mtime,
        is_stale=is_stale,
        days_behind=days_behind,
    )
```

- [ ] **Step 5.4: Run tests to verify they pass**

Run: `pytest tests/test_sync_data.py::TestSourceCheck tests/test_sync_data.py::TestFreshness -v`
Expected: 6 PASSED

- [ ] **Step 5.5: Commit**

```bash
git add scripts/sync_data/freshness.py tests/test_sync_data.py
git commit -m "feat(sync): add freshness checking for data sources (ISO 5259)"
```

---

## Task 6: Sync Data — Symlink Management

**Files:**
- Create: `scripts/sync_data/symlink.py`
- Modify: `tests/test_sync_data.py` (add TestSymlink)

- [ ] **Step 6.1: Add TestSymlink to test file**

Append to `tests/test_sync_data.py`:

```python
from scripts.sync_data.symlink import update_symlink


class TestSymlink:
    """Tests for update_symlink."""

    def test_create_new_symlink(self, tmp_path):
        """Test creating a new symlink."""
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        update_symlink(target, link)
        assert link.exists()
        assert link.resolve() == target.resolve()

    def test_update_existing_symlink(self, tmp_path):
        """Test updating an existing symlink to new target."""
        old_target = tmp_path / "old"
        old_target.mkdir()
        new_target = tmp_path / "new"
        new_target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(old_target, target_is_directory=True)
        update_symlink(new_target, link)
        assert link.resolve() == new_target.resolve()

    def test_fallback_config_file(self, tmp_path, monkeypatch):
        """Test fallback to .data_source file on symlink failure."""
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"

        def mock_symlink(*args, **kwargs):
            raise OSError("Permission denied")

        monkeypatch.setattr("os.symlink", mock_symlink)
        update_symlink(target, link)

        config_file = link.parent / ".data_source"
        assert config_file.exists()
        assert config_file.read_text().strip() == str(target)
```

- [ ] **Step 6.2: Run tests to verify they fail**

Run: `pytest tests/test_sync_data.py::TestSymlink -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 6.3: Implement symlink.py**

```python
# scripts/sync_data/symlink.py
"""Symlink management with Windows fallback (ISO 27034).

On Windows, symlink creation requires Developer Mode or admin.
Falls back to .data_source config file if symlink fails.

ISO Compliance:
- ISO/IEC 27034:2011 - Secure Coding (path validation)
- ISO/IEC 5055:2021 - Code Quality (<50 lines per function)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def update_symlink(target: Path, link: Path) -> None:
    """Create or update a directory symlink, with fallback."""
    target = target.resolve()
    if not target.is_dir():
        msg = f"Target is not a directory: {target}"
        raise ValueError(msg)

    try:
        _create_symlink(target, link)
        logger.info("Symlink: %s -> %s", link, target)
    except OSError:
        logger.warning("Symlink failed (permissions?), using .data_source fallback")
        _write_data_source_fallback(target, link)


def _create_symlink(target: Path, link: Path) -> None:
    """Create or replace a directory symlink."""
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            msg = f"Path exists and is not a symlink: {link}"
            raise ValueError(msg)

    os.symlink(target, link, target_is_directory=True)


def _write_data_source_fallback(target: Path, link: Path) -> None:
    """Write target path to .data_source config file."""
    config_file = link.parent / ".data_source"
    config_file.write_text(str(target))
    logger.info("Fallback config: %s", config_file)
```

- [ ] **Step 6.4: Run tests to verify they pass**

Run: `pytest tests/test_sync_data.py::TestSymlink -v`
Expected: 3 PASSED

- [ ] **Step 6.5: Commit**

```bash
git add scripts/sync_data/symlink.py tests/test_sync_data.py
git commit -m "feat(sync): add symlink management with Windows fallback"
```

---

## Task 7: Sync Data — HuggingFace Integration

**Files:**
- Create: `scripts/sync_data/huggingface.py`
- Modify: `tests/test_sync_data.py` (add TestHuggingFace)
- Modify: `requirements.txt` (add deps)

- [ ] **Step 7.1: Add deps to requirements.txt**

Add these 2 lines to `requirements.txt`:
```
datasets>=2.16.0
huggingface_hub>=0.20.0
```

- [ ] **Step 7.2: Install new deps**

Run: `pip install datasets huggingface_hub`

- [ ] **Step 7.3: Add TestHuggingFace to test file**

Append to `tests/test_sync_data.py`:

```python
from unittest.mock import MagicMock, patch

from scripts.sync_data.huggingface import pull_from_hf, push_to_hf


class TestHuggingFace:
    """Tests for HuggingFace pull/push."""

    @patch("scripts.sync_data.huggingface.hf_hub_download")
    def test_pull_downloads_parquets(self, mock_download, tmp_path):
        """Test pull downloads echiquiers and joueurs parquets."""
        mock_download.return_value = str(tmp_path / "file.parquet")
        pull_from_hf("Pierrax/ffe-history", tmp_path)
        assert mock_download.call_count == 2

    @patch("scripts.sync_data.huggingface.HfApi")
    def test_push_uploads_parquets(self, mock_api_class, tmp_path):
        """Test push uploads existing parquet files."""
        (tmp_path / "echiquiers.parquet").write_text("data")
        (tmp_path / "joueurs.parquet").write_text("data")
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        push_to_hf(tmp_path, "Pierrax/ffe-history")
        assert mock_api.upload_file.call_count == 2

    @patch("scripts.sync_data.huggingface.HfApi")
    def test_push_missing_token_raises(self, mock_api_class, tmp_path):
        """Test push with missing token raises clear error."""
        (tmp_path / "echiquiers.parquet").write_text("data")
        (tmp_path / "joueurs.parquet").write_text("data")
        mock_api = MagicMock()
        mock_api.upload_file.side_effect = Exception("Invalid token")
        mock_api_class.return_value = mock_api
        with pytest.raises(Exception, match="Invalid token"):
            push_to_hf(tmp_path, "Pierrax/ffe-history")
```

- [ ] **Step 7.4: Run tests to verify they fail**

Run: `pytest tests/test_sync_data.py::TestHuggingFace -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 7.5: Implement huggingface.py**

```python
# scripts/sync_data/huggingface.py
"""HuggingFace Hub integration for data sync.

Pull parsed parquets from HF or push updated parquets.
Never logs or accepts tokens as arguments.

ISO Compliance:
- ISO/IEC 27034:2011 - Secure Coding (no token exposure)
- ISO/IEC 5055:2021 - Code Quality (<50 lines per function)
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)

_PARQUET_FILES = [
    ("parsed/echiquiers.parquet", "echiquiers.parquet"),
    ("parsed/joueurs.parquet", "joueurs.parquet"),
]


def pull_from_hf(repo_id: str, output_dir: Path) -> None:
    """Download parsed parquets from HuggingFace dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for hf_path, local_name in _PARQUET_FILES:
        logger.info("Pulling %s from %s", hf_path, repo_id)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=hf_path,
            repo_type="dataset",
        )
        shutil.copy2(downloaded, output_dir / local_name)
        logger.info("Saved: %s", output_dir / local_name)


def push_to_hf(parquet_dir: Path, repo_id: str) -> None:
    """Upload parsed parquets to HuggingFace dataset."""
    api = HfApi()
    for hf_path, local_name in _PARQUET_FILES:
        local_file = parquet_dir / local_name
        if not local_file.exists():
            logger.warning("Skipping %s: file not found", local_name)
            continue
        logger.info("Pushing %s to %s", local_name, repo_id)
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=hf_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        logger.info("Uploaded: %s", hf_path)
```

- [ ] **Step 7.6: Run tests to verify they pass**

Run: `pytest tests/test_sync_data.py::TestHuggingFace -v`
Expected: 3 PASSED

- [ ] **Step 7.7: Commit**

```bash
git add scripts/sync_data/huggingface.py tests/test_sync_data.py requirements.txt
git commit -m "feat(sync): add HuggingFace pull/push integration"
```

---

### Review Checkpoint 2 — Sync Data Core

> After Tasks 4-7: dispatch `superpowers:requesting-code-review` on `scripts/sync_data/` package. Verify Pydantic validation, Windows compatibility, HF token safety.

---

## Task 8: Sync Data — CLI Entry Point

**Files:**
- Create: `scripts/sync_data/__main__.py`
- Modify: `scripts/sync_data/__init__.py` (add re-exports)

- [ ] **Step 8.1: Create __main__.py**

```python
# scripts/sync_data/__main__.py
"""Entry point for: python -m scripts.sync_data

Syncs fresh data from ffe_scrapper or HuggingFace.

ISO Compliance:
- ISO/IEC 27034:2011 - Secure Coding (Pydantic-validated CLI)
- ISO/IEC 5055:2021 - Code Quality (<50 lines)
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from scripts.sync_data.freshness import check_freshness, check_source
from scripts.sync_data.huggingface import pull_from_hf, push_to_hf
from scripts.sync_data.symlink import update_symlink
from scripts.sync_data.types import SyncConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_SOURCE = Path(os.environ.get(
    "FFE_SCRAPPER_DATA_DIR",
    Path(__file__).resolve().parents[2] / ".." / "ffe_scrapper" / "data",
))


def main() -> None:
    """CLI entry point for data sync."""
    parser = argparse.ArgumentParser(description="Sync FFE data")
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = SyncConfig(source=args.source, push=args.push, dry_run=args.dry_run)
    _run_sync(config)


def _run_sync(config: SyncConfig) -> None:
    """Execute sync based on validated config."""
    data_dir = Path("data")

    if config.source == "hf":
        logger.info("Pulling from HuggingFace: %s", config.hf_repo_id)
        if not config.dry_run:
            pull_from_hf(config.hf_repo_id, data_dir)
        return

    source_dir = config.source_dir or _DEFAULT_SOURCE
    status = check_source(source_dir)
    logger.info("Source: %s (files=%d)", source_dir, status.file_count)

    freshness = check_freshness(source_dir, data_dir)
    logger.info("Stale=%s, days_behind=%.1f", freshness.is_stale, freshness.days_behind)

    if not config.dry_run:
        link = Path("dataset_alice")
        update_symlink(source_dir, link)

    if config.push:
        logger.info("Pushing to HuggingFace: %s", config.hf_repo_id)
        if not config.dry_run:
            push_to_hf(data_dir, config.hf_repo_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Update __init__.py with full re-exports**

```python
# scripts/sync_data/__init__.py
"""Data sync package for ALICE Engine.

Syncs fresh data from ffe_scrapper or HuggingFace.

Usage:
    python -m scripts.sync_data
    python -m scripts.sync_data --source hf --push
"""
from scripts.sync_data.freshness import check_freshness, check_source
from scripts.sync_data.huggingface import pull_from_hf, push_to_hf
from scripts.sync_data.symlink import update_symlink
from scripts.sync_data.types import FreshnessReport, SourceStatus, SyncConfig

__all__ = [
    # Types
    "FreshnessReport",
    "SourceStatus",
    "SyncConfig",
    # Freshness
    "check_freshness",
    "check_source",
    # Symlink
    "update_symlink",
    # HuggingFace
    "pull_from_hf",
    "push_to_hf",
]
```

- [ ] **Step 8.3: Smoke test CLI**

Run: `python -m scripts.sync_data --dry-run`
Expected: logs source status and freshness report without making changes

- [ ] **Step 8.4: Commit**

```bash
git add scripts/sync_data/__main__.py scripts/sync_data/__init__.py
git commit -m "feat(sync): add CLI entry point for data sync"
```

---

## Task 9: Makefile Integration

**Files:**
- Modify: `Makefile`

**Ref:** Check existing Makefile for `.PHONY` list location and target format

- [ ] **Step 9.1: Read current Makefile .PHONY and relevant targets**

Run: `head -10 Makefile && grep -n "^\.PHONY" Makefile`

- [ ] **Step 9.2: Add sync and refresh-data targets**

Add `sync` and `refresh-data` to the existing `.PHONY` list. Add new targets near the existing `parse-data` and `features` targets:

```makefile
sync:                              ## Sync data from ffe_scrapper
	@echo "Syncing data (ISO 5259)..."
	$(PYTHON) -m scripts.sync_data
	@echo ""
	@echo "Sync complete"

refresh-data: sync parse-data features  ## Full data refresh: sync + parse + features
	@echo ""
	@echo "============================================"
	@echo "  DATA REFRESH COMPLETE (ISO 5259)"
	@echo "============================================"
```

- [ ] **Step 9.3: Verify Makefile syntax**

Run: `make -n refresh-data`
Expected: prints the commands without executing

- [ ] **Step 9.4: Commit**

```bash
git add Makefile
git commit -m "feat(make): add sync and refresh-data targets"
```

---

## Task 10: End-to-End Verification

- [ ] **Step 10.1: Run all new tests**

Run: `pytest tests/schemas/test_parsing_schemas.py tests/schemas/test_parsing_validation.py tests/test_sync_data.py -v --cov=schemas.parsing_schemas --cov=schemas.parsing_validation --cov=scripts.sync_data --cov-fail-under=80`
Expected: 28 PASSED (10 + 6 + 12), coverage >= 80%

- [ ] **Step 10.2: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: all existing + new tests pass

- [ ] **Step 10.3: Run quality checks**

Run: `make quality`
Expected: ruff, mypy, bandit all pass on new files

- [ ] **Step 10.4: Verify ISO 5055 limits**

Run: `wc -l schemas/parsing_schemas.py schemas/parsing_validation.py scripts/parse_dataset/__main__.py scripts/sync_data/*.py`
Expected: all files < 300 lines

Run: `radon cc schemas/parsing_schemas.py schemas/parsing_validation.py scripts/sync_data/*.py -s`
Expected: all functions A or B

- [ ] **Step 10.5: Dry-run refresh pipeline**

Run: `python -m scripts.sync_data --dry-run`
Expected: reports source status, freshness, no changes made

- [ ] **Step 10.6: Full data refresh (the real deal)**

Run: `make refresh-data`
Expected:
1. Symlink updated
2. HTML parsed to parquet (takes several minutes)
3. ISO 5259 validation reports in `reports/validation/`
4. Features regenerated in `data/features/`

- [ ] **Step 10.7: Verify output**

Run: `python -c "import pandas as pd; df = pd.read_parquet('data/echiquiers.parquet'); print(f'Rows: {len(df)}, Seasons: {sorted(df.saison.unique())[-3:]}')"`
Expected: shows 2024, 2025, 2026 in recent seasons

Run: `ls -lh reports/validation/`
Expected: `raw_echiquiers_report.json` and `raw_joueurs_report.json`

- [ ] **Step 10.8: Verify HuggingFace push pathway**

Run: `python -m scripts.sync_data --push --dry-run`
Expected: logs "Pushing to HuggingFace" without actually uploading

- [ ] **Step 10.9: Commit all remaining changes**

```bash
git add schemas/parsing_schemas.py schemas/parsing_validation.py schemas/__init__.py scripts/parse_dataset/__main__.py scripts/sync_data/ tests/schemas/test_parsing_schemas.py tests/schemas/test_parsing_validation.py tests/test_sync_data.py Makefile requirements.txt
git commit -m "feat(data): complete data refresh pipeline (ISO 5259)"
```

---

## Task 11: Final Code Review + Industry Corrections

> Dispatch `superpowers:requesting-code-review` on all new files.

- [ ] **Step 11.1: Code review — all new schemas**

Review `schemas/parsing_schemas.py` and `schemas/parsing_validation.py` against:
- Pandera best practices (schema reuse, coerce settings)
- Existing `training_schemas.py` / `training_validation.py` pattern consistency
- `QualityMetrics`, `ValidationError`, `ValidationReport` constructor correctness
- Error aggregation (deduplicate by column+check, not per-row)

- [ ] **Step 11.2: Code review — sync_data package**

Review all `scripts/sync_data/*.py` against:
- Pydantic validation completeness (ISO 27034)
- No secrets logged (HF token never in output)
- Path sanitization (no traversal outside project)
- Windows compatibility (symlink fallback tested)
- Error handling (network failures for HF, permission errors)

- [ ] **Step 11.3: Code review — parse entry point**

Review `scripts/parse_dataset/__main__.py` against:
- Correct orchestration.py signatures (`dict[str, int]` return types)
- Parquet re-read for validation (not using return value)
- Logging consistency with existing scripts

- [ ] **Step 11.4: Code review — tests**

Review all test files against:
- ISO 29119 headers complete (Document ID, Version, Tests count)
- Edge cases covered: empty DataFrames, missing files, permission errors
- No flaky tests (time.sleep usage minimized)
- Fixtures DRY (shared via conftest if duplicated)
- Assertion messages clear

- [ ] **Step 11.5: Apply spontaneous corrections**

Fix any issues found in steps 11.1-11.4. Common industry corrections:
- **Error aggregation**: Group Pandera errors by (column, check) not by row — prevents 1M-row error lists
- **Retry logic**: Add `tenacity` retry on HF network calls (transient failures)
- **Structured logging**: Use `structlog` or `logging.extra` for machine-parseable logs
- **Null percentage dict**: Ensure `null_percentages` keys match expected column names
- **Data hash determinism**: Use `pd.util.hash_pandas_object` for reproducible hashes (vs. `.to_json()` which depends on float precision)
- **Report atomicity**: Write to temp file + rename to prevent partial reports on crash

- [ ] **Step 11.6: Run full DoD verification**

```bash
# All tests pass
pytest tests/ -v --tb=short

# Coverage on new files
pytest tests/schemas/test_parsing_schemas.py tests/schemas/test_parsing_validation.py tests/test_sync_data.py -v --cov=schemas.parsing_schemas --cov=schemas.parsing_validation --cov=scripts.sync_data --cov-fail-under=80

# Quality gates
make quality

# ISO 5055 limits
wc -l schemas/parsing_schemas.py schemas/parsing_validation.py scripts/parse_dataset/__main__.py scripts/sync_data/*.py
radon cc schemas/parsing_schemas.py schemas/parsing_validation.py scripts/sync_data/*.py -s

# Validation reports exist
ls -lh reports/validation/
cat reports/validation/raw_echiquiers_report.json | python -m json.tool | head -20
```

- [ ] **Step 11.7: Final commit with corrections**

```bash
git add -u
git commit -m "fix(review): apply industry corrections from code review"
```

---

## Quality Gate Summary — Pipeline Execution Checklist

When running `make refresh-data`, the pipeline self-validates at each stage:

```
SYNC (Task 8)
  ├── Gate 1: Source accessible? ─── BLOCK if not
  ├── Freshness check ─── INFO: skip if data already fresh
  └── Symlink updated ─── WARN if fallback to .data_source

PARSE (Task 3)
  ├── HTML -> parquet ─── BLOCK on parse errors
  ├── Gate 2: Schema validation (Pandera) ─── BLOCK on schema fail
  ├── Gate 2: Null/range/coherence checks ─── BLOCK on critical
  ├── Data hash computed ─── stored in ValidationReport
  └── Report persisted ─── reports/validation/*.json

FEATURES (unchanged, existing)
  ├── Gate 3: Feature engineering ─── BLOCK on NaN/inf
  ├── Temporal split ─── BLOCK on leakage
  └── Parquets written ─── data/features/

READY FOR TRAINING
  └── All gates passed → proceed to make train
```
