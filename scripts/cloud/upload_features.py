"""Upload feature parquets to Kaggle Dataset.

One-time helper to upload pre-computed features as a Kaggle Dataset.
The Kaggle training script reads from /kaggle/input/alice-features/.

Usage: python -m scripts.cloud.upload_features
Requires: ~/.kaggle/kaggle.json or KAGGLE_API_TOKEN env var

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (<50 lines)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")
DATASET_SLUG = "pguillemin/alice-features"
FILES = ["train.parquet", "valid.parquet", "test.parquet"]


def upload() -> None:
    """Upload feature parquets as Kaggle Dataset via Python API."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    for f in FILES:
        if not (FEATURES_DIR / f).exists():
            msg = f"Missing: {FEATURES_DIR / f}. Run 'make refresh-data' first."
            raise FileNotFoundError(msg)

    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for f in FILES:
            shutil.copy2(FEATURES_DIR / f, tmp_path / f)
            logger.info("Copied %s (%s MB)", f, f"{(FEATURES_DIR / f).stat().st_size / 1e6:.1f}")

        # Kaggle API requires dataset-metadata.json in the folder
        import json

        meta = {
            "title": "ALICE Engine Features",
            "id": DATASET_SLUG,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (tmp_path / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

        logger.info("Uploading to Kaggle as %s...", DATASET_SLUG)
        try:
            api.dataset_create_new(folder=str(tmp_path), dir_mode="zip", public=False)
            logger.info("Created new dataset: %s", DATASET_SLUG)
        except Exception:
            logger.info("Dataset exists, creating new version...")
            api.dataset_create_version(
                folder=str(tmp_path),
                version_notes="Feature refresh",
                dir_mode="zip",
            )
            logger.info("Version updated: %s", DATASET_SLUG)


if __name__ == "__main__":
    upload()
