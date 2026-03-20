# Data Refresh Pipeline — Design Spec

**Date**: 2026-03-17
**Status**: Approved
**Approach**: B (Integrated — sync + validation ISO 5259)
**ISO Compliance**: 5055 (SRP, limits), 27034 (Pydantic), 29119 (tests), 5259 (lineage), 42001 (traceability)

## Problem

The training pipeline works end-to-end but has 4 gaps preventing reproducible data refresh:

1. `dataset_alice/` symlink points to `ffe_data_backup/` (stale) instead of `ffe_scrapper/data/` (fresh, updated 2026-03-17)
2. `make parse-data` is broken — `scripts/parse_dataset/__main__.py` is missing
3. No HuggingFace integration (`datasets`/`huggingface_hub` not in requirements, no sync script)
4. ISO 5259 validation (Pandera) exists in `schemas/training_validation.py` but is not wired into the parsing pipeline

## Goal

A single `make refresh-data` command that: syncs fresh HTML from ffe_scrapper, parses to parquet, validates ISO 5259, generates features — ready for training.

## Data Flow

```
ffe_scrapper/data/                  (HTML bruts, source de verite)
    |                                Layout: {year}/{competition}/{division}/{groupe}/ronde_N.html
    v                                         players_v2/{club}/page_N.html
scripts/sync_data/                  (NEW package — freshness check, symlink, HF pull/push)
    |
    v
dataset_alice/                      (symlink -> ffe_scrapper/data/)
    |
    v
scripts/parse_dataset/__main__.py   (NEW — CLI entry point)
    |
    v
data/echiquiers.parquet             (re-parsed, ~1.7M rows, 32 columns)
data/joueurs.parquet                (re-parsed, ~66k rows, 19 columns)
    |
    v
schemas/parsing_schemas.py          (NEW — Pandera schemas for raw parquets)
schemas/parsing_validation.py       (NEW — validation + reporting, ISO 5259)
    |                                Reports persisted to reports/validation/
    v
scripts/feature_engineering.py      (unchanged)
    |
    v
data/features/{train,valid,test}.parquet
```

## Constraints (ISO)

All new code MUST respect:
- **ISO 5055**: max 300 lines/file, max 50 lines/function, complexity <= B (Xenon), 1 file = 1 responsibility
- **ISO 27034**: Pydantic for ALL input validation, no hardcoded secrets, path sanitization
- **ISO 29119**: structured docstrings (Document ID, Version, Tests count), fixtures, thematic test classes
- **ISO 5259**: data lineage (source hash, timestamps), validation reports persisted to disk
- **ISO 15289**: module docstrings with ISO compliance references

## Components

### 1. `scripts/sync_data/` (NEW package, SRP split)

Following the `scripts/parse_dataset/` and `schemas/training_*` pattern, split into:

| File | Responsibility | Est. lines |
|------|----------------|-----------|
| `__init__.py` | Re-exports | ~15 |
| `__main__.py` | CLI entry point (argparse → SyncConfig) | ~35 |
| `types.py` | Pydantic models: `SyncConfig`, `SourceStatus`, `FreshnessReport` | ~50 |
| `freshness.py` | `check_source()`, `check_freshness()` | ~60 |
| `symlink.py` | `update_symlink()` with Windows fallback | ~40 |
| `huggingface.py` | `pull_from_hf()`, `push_to_hf()` (never logs tokens) | ~60 |

CLI interface:
```
python -m scripts.sync_data                    # Default: sync from local ffe_scrapper
python -m scripts.sync_data --source hf        # Pull from HuggingFace instead
python -m scripts.sync_data --push             # Push parquets to HF after sync
python -m scripts.sync_data --dry-run          # Report what would change
```

**`types.py`** — Pydantic models:
```python
class SyncConfig(BaseModel):
    """Validated CLI configuration (ISO 27034)."""
    source: Literal["local", "hf"] = "local"
    source_dir: DirectoryPath = Field(default=None)  # Resolved from env or CLI
    push: bool = False
    dry_run: bool = False
    hf_repo_id: str = Field(pattern=r"^[\w-]+/[\w.-]+$", default="Pierrax/ffe-history")
```

Configuration: `FFE_SCRAPPER_DATA` from env var `FFE_SCRAPPER_DATA_DIR`, fallback to project-relative resolution. No hardcoded absolute paths in source code.

Platform note: On Windows, `os.symlink(target, link, target_is_directory=True)` requires Developer Mode or admin rights. If symlink creation fails, fallback to writing the data path to a `.data_source` config file that `parse_dataset` reads.

### 2. `scripts/parse_dataset/__main__.py` (NEW, ~40 lines)

Wrapper CLI around existing `orchestration.parse_compositions()` and `orchestration.parse_joueurs()`.

Note: `parse_compositions()` and `parse_joueurs()` return `dict[str, int]` (stats), NOT DataFrames. The `__main__.py` re-reads the output parquets for validation.

```python
"""Entry point for: python -m scripts.parse_dataset

ISO 5259 compliant parsing with post-parse validation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from scripts.parse_dataset.orchestration import parse_compositions, parse_joueurs
from schemas.parsing_validation import validate_raw_echiquiers, validate_raw_joueurs

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("dataset_alice")
    output_dir = Path("data")

    logger.info("Parsing compositions -> echiquiers.parquet")
    stats_ech = parse_compositions(data_dir, output_dir / "echiquiers.parquet")
    logger.info("Compositions parsed: %s", stats_ech)

    logger.info("Parsing players -> joueurs.parquet")
    stats_jou = parse_joueurs(data_dir / "players_v2", output_dir / "joueurs.parquet")
    logger.info("Players parsed: %s", stats_jou)

    # ISO 5259: validate raw parquets + persist reports
    logger.info("Validating ISO 5259 (raw parsing schemas)")
    df_ech = pd.read_parquet(output_dir / "echiquiers.parquet")
    df_jou = pd.read_parquet(output_dir / "joueurs.parquet")
    validate_raw_echiquiers(df_ech, source_path=str(output_dir / "echiquiers.parquet"))
    validate_raw_joueurs(df_jou, source_path=str(output_dir / "joueurs.parquet"))

    logger.info("Parsing + validation complete")


if __name__ == "__main__":
    main()
```

### 3. `schemas/parsing_schemas.py` (NEW, ~80 lines)

Pandera schemas ONLY for raw parsed parquets. Follows existing `training_schemas.py` pattern (schemas only, no orchestration).

**EchiquiersRawSchema** validates:
- Required columns: saison, competition, division, groupe, ronde, echiquier, blanc_nom, noir_nom, blanc_elo, noir_elo, resultat_blanc, resultat_noir, type_resultat, diff_elo
- saison in [2002, 2030]
- echiquier in [1, 12]
- ronde in [1, 15]
- blanc_elo / noir_elo in [0, 3000]
- resultat_blanc / resultat_noir in {0.0, 0.5, 1.0}
- diff_elo == blanc_elo - noir_elo (coherence check)

**JoueursRawSchema** validates:
- Required columns: nr_ffe, nom, prenom, elo, elo_type, categorie, club
- elo in [0, 3000]
- elo_type in {"F", "N", "E"}
- nr_ffe matches pattern `^[A-Z]\d+$`

### 4. `schemas/parsing_validation.py` (NEW, ~80 lines)

Validation orchestration + report persistence. Follows existing `training_validation.py` pattern. Reuses `DataLineage`, `QualityMetrics`, `ValidationReport` from `schemas/training_types.py`.

Functions:
- `validate_raw_echiquiers(df, source_path) -> ValidationReport` — validate + persist to `reports/validation/raw_echiquiers_report.json`
- `validate_raw_joueurs(df, source_path) -> ValidationReport` — validate + persist to `reports/validation/raw_joueurs_report.json`

Each report includes: DataLineage (source_path, source_hash, row_count, validation_timestamp), QualityMetrics (null_percentage, duplicates, value ranges), schema errors if any.

### 5. Makefile targets (MODIFY)

```makefile
.PHONY: sync refresh-data

sync:                              ## Sync data from ffe_scrapper
	python -m scripts.sync_data

refresh-data: sync parse-data features  ## Full data refresh: sync + parse + features
	@echo "Data refresh complete"

# Existing target, now works with __main__.py:
parse-data:                        ## Parse HTML -> parquets
	python -m scripts.parse_dataset
```

### 6. Requirements (MODIFY)

Add to `requirements.txt`:
```
datasets>=2.16.0
huggingface_hub>=0.20.0
```

## What does NOT change

- `scripts/feature_engineering.py` — consumes same parquets, same interface
- `scripts/train_models_parallel.py` — consumes same feature parquets
- `scripts/autogluon/` — unchanged
- All ISO validation scripts (robustness, fairness, mcnemar, impact) — unchanged
- All existing tests — unchanged
- `services/data_loader.py` — unchanged
- `schemas/training_validation.py` — unchanged (validates post-feature data, different stage)
- `schemas/training_schemas.py` — unchanged (feature-level Pandera schemas)
- `schemas/training_types.py` — unchanged (reused by new parsing_validation.py)
- `scripts/parse_dataset/orchestration.py` — unchanged (validation lives in __main__.py)

## Test Plan (ISO 29119)

### `tests/test_sync_data.py` (~150 lines)

```
Document ID: TEST-SYNC-001
Version: 1.0
Tests: ~12
```

| Class | Tests | Description |
|-------|-------|-------------|
| `TestSourceCheck` | 3 | accessible dir, missing dir, empty dir |
| `TestFreshness` | 3 | stale data, fresh data, no parquets yet |
| `TestSymlink` | 3 | create new, update existing, Windows fallback |
| `TestHuggingFace` | 3 | pull (mocked), push (mocked), missing token error |

Fixtures: `tmp_source_dir`, `tmp_parquet_dir`, `mock_hf_api`

### `tests/schemas/test_parsing_schemas.py` (~120 lines)

```
Document ID: TEST-PARSE-SCHEMA-001
Version: 1.0
Tests: ~10
```

| Class | Tests | Description |
|-------|-------|-------------|
| `TestEchiquiersRawSchema` | 5 | valid df, invalid elo, invalid saison, missing columns, diff_elo coherence |
| `TestJoueursRawSchema` | 5 | valid df, invalid elo_type, invalid nr_ffe pattern, missing columns, null handling |

Fixtures: `sample_echiquiers_df`, `sample_joueurs_df` (reuse pattern from `conftest.py`)

### `tests/schemas/test_parsing_validation.py` (~80 lines)

```
Document ID: TEST-PARSE-VALID-001
Version: 1.0
Tests: ~6
```

| Class | Tests | Description |
|-------|-------|-------------|
| `TestValidateRawEchiquiers` | 3 | valid report, report persistence, critical error raises |
| `TestValidateRawJoueurs` | 3 | valid report, report persistence, critical error raises |

Fixtures: reuse `sample_echiquiers_df`, `sample_joueurs_df`, `tmp_path`

**Coverage target**: >80% on all new files

## File inventory

| Action | File | Est. lines |
|--------|------|-----------|
| CREATE | `scripts/sync_data/__init__.py` | ~15 |
| CREATE | `scripts/sync_data/__main__.py` | ~35 |
| CREATE | `scripts/sync_data/types.py` | ~50 |
| CREATE | `scripts/sync_data/freshness.py` | ~60 |
| CREATE | `scripts/sync_data/symlink.py` | ~40 |
| CREATE | `scripts/sync_data/huggingface.py` | ~60 |
| CREATE | `scripts/parse_dataset/__main__.py` | ~40 |
| CREATE | `schemas/parsing_schemas.py` | ~80 |
| CREATE | `schemas/parsing_validation.py` | ~80 |
| CREATE | `tests/test_sync_data.py` | ~150 |
| CREATE | `tests/schemas/test_parsing_schemas.py` | ~120 |
| CREATE | `tests/schemas/test_parsing_validation.py` | ~80 |
| MODIFY | `Makefile` | +8 |
| MODIFY | `requirements.txt` | +2 |
| MODIFY | `CLAUDE.md` | already done |

**Total**: 12 new files (~810 lines), 2 modified files

## Success criteria

1. `make refresh-data` runs end-to-end without error
2. `data/echiquiers.parquet` contains 2026 season data (post-March 17)
3. ISO 5259 validation reports persisted to `reports/validation/` with no critical errors
4. `data/features/` regenerated with updated temporal splits
5. `python -m scripts.sync_data --push` uploads parquets to HuggingFace
6. All new tests pass with >80% coverage
7. All files <300 lines, all functions <50 lines, complexity <= B
