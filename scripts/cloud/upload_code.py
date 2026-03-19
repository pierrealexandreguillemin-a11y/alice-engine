"""Upload feature engineering code to Kaggle Dataset.

Packages scripts/features/ + scripts/feature_engineering.py as a Kaggle
Dataset so the Kaggle training script can import and run feature engineering
from raw parquets — no pre-computed features needed.

Usage: python -m scripts.cloud.upload_code
Requires: KAGGLE_API_TOKEN env var or ~/.kaggle/kaggle.json

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (<50 lines per function)
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_SLUG = "pguillemin/alice-code"


def upload() -> None:
    """Package and upload feature engineering code as Kaggle Dataset."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _package_code(tmp_path)
        _write_metadata(tmp_path)
        _push_to_kaggle(api, tmp_path)


def _package_code(tmp_path: Path) -> None:
    """Copy feature engineering code to temp directory."""
    # scripts/features/ (all modules)
    src_features = PROJECT_ROOT / "scripts" / "features"
    dst_features = tmp_path / "scripts" / "features"
    shutil.copytree(src_features, dst_features, ignore=shutil.ignore_patterns("__pycache__"))

    # scripts/feature_engineering.py
    shutil.copy2(
        PROJECT_ROOT / "scripts" / "feature_engineering.py",
        tmp_path / "scripts" / "feature_engineering.py",
    )

    # scripts/__init__.py (needed for imports)
    (tmp_path / "scripts" / "__init__.py").write_text('"""Scripts package."""\n')

    # schemas/ (needed for training_constants)
    src_schemas = PROJECT_ROOT / "schemas"
    dst_schemas = tmp_path / "schemas"
    shutil.copytree(src_schemas, dst_schemas, ignore=shutil.ignore_patterns("__pycache__"))

    # data/joueurs.parquet (needed for player enrichment)
    joueurs = PROJECT_ROOT / "data" / "joueurs.parquet"
    if joueurs.exists():
        (tmp_path / "data").mkdir()
        shutil.copy2(joueurs, tmp_path / "data" / "joueurs.parquet")

    logger.info(
        "Packaged code: %s", [str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.py")][:10]
    )


def _write_metadata(tmp_path: Path) -> None:
    """Write Kaggle dataset metadata."""
    meta = {"title": "ALICE Engine Code", "id": DATASET_SLUG, "licenses": [{"name": "CC0-1.0"}]}
    (tmp_path / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


def _push_to_kaggle(api: object, tmp_path: Path) -> None:
    """Create or update dataset on Kaggle."""
    logger.info("Uploading code to Kaggle as %s...", DATASET_SLUG)
    try:
        api.dataset_create_new(folder=str(tmp_path), dir_mode="zip", public=False)
        logger.info("Created new dataset: %s", DATASET_SLUG)
    except Exception:
        logger.info("Dataset exists, creating new version...")
        api.dataset_create_version(
            folder=str(tmp_path), version_notes="Code update", dir_mode="zip"
        )
        logger.info("Version updated: %s", DATASET_SLUG)


if __name__ == "__main__":
    upload()
