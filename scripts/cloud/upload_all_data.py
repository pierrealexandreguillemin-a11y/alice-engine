"""Upload all data + code to Kaggle Dataset pguillemin/alice-code.

Everything the Kaggle kernel needs in one dataset:
- data/echiquiers.parquet (raw parsed games)
- data/joueurs.parquet (player registry)
- scripts/ (feature engineering code)
- schemas/ (validation constants)

Usage: python -m scripts.cloud.upload_all_data
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
    """Package everything and upload to Kaggle."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _package_all(tmp_path)
        meta = {
            "title": "alice-code",
            "id": DATASET_SLUG,
            "licenses": [{"name": "CC0-1.0"}],
            "isPrivate": True,
        }
        (tmp_path / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

        logger.info("Uploading to Kaggle as %s...", DATASET_SLUG)
        try:
            api.dataset_create_version(
                folder=str(tmp_path),
                version_notes="Full data + code",
                dir_mode="zip",
            )
            logger.info("Version updated: %s", DATASET_SLUG)
        except Exception:
            api.dataset_create_new(folder=str(tmp_path), dir_mode="zip", public=False)
            logger.info("Created: %s", DATASET_SLUG)


def _package_all(tmp_path: Path) -> None:
    """Copy data + code to temp directory."""
    # Data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy2(PROJECT_ROOT / "data" / "echiquiers.parquet", data_dir / "echiquiers.parquet")
    shutil.copy2(PROJECT_ROOT / "data" / "joueurs.parquet", data_dir / "joueurs.parquet")
    logger.info("Copied echiquiers.parquet + joueurs.parquet")

    # Scripts — copy EVERYTHING (all modules needed for feature engineering)
    src_scripts = PROJECT_ROOT / "scripts"
    dst_scripts = tmp_path / "scripts"
    shutil.copytree(
        src_scripts,
        dst_scripts,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "cloud",
            "autogluon",
            "comparison",
            "reports",
            "model_registry",
            "training",
            "ensemble*",
            "monitoring",
            "alerts",
            "agents",
            "calibration",
            "evaluation",
            "robustness",
            "fairness",
            "ml_types",
            "train_models*",
        ),
    )
    logger.info("Copied scripts/ (feature engineering + dependencies)")

    # AutoGluon diagnostics (needed by train_autogluon_kaggle.py)
    cloud_dir = dst_scripts / "cloud"
    cloud_dir.mkdir(exist_ok=True)
    (cloud_dir / "__init__.py").touch()
    shutil.copy2(
        src_scripts / "cloud" / "autogluon_diagnostics.py",
        cloud_dir / "autogluon_diagnostics.py",
    )
    logger.info("Copied autogluon_diagnostics.py")

    # Schemas
    shutil.copytree(
        PROJECT_ROOT / "schemas", tmp_path / "schemas", ignore=shutil.ignore_patterns("__pycache__")
    )
    logger.info("Copied schemas/")

    total = sum(1 for _ in tmp_path.rglob("*") if _.is_file())
    logger.info("Total files: %d", total)


if __name__ == "__main__":
    upload()
