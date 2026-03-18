"""Upload feature parquets to Kaggle Dataset.

One-time helper to upload pre-computed features as a Kaggle Dataset.
The Kaggle training script reads from /kaggle/input/alice-features/.

Usage: python -m scripts.cloud.upload_features
Requires: ~/.kaggle/kaggle.json with API credentials

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (<50 lines)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")
DATASET_ID = "pierrax/alice-features"
FILES = ["train.parquet", "valid.parquet", "test.parquet"]


def upload() -> None:
    """Upload feature parquets as Kaggle Dataset."""
    for f in FILES:
        if not (FEATURES_DIR / f).exists():
            msg = f"Missing: {FEATURES_DIR / f}. Run 'make refresh-data' first."
            raise FileNotFoundError(msg)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for f in FILES:
            shutil.copy2(FEATURES_DIR / f, tmp_path / f)

        metadata = {
            "title": "ALICE Engine Features",
            "id": DATASET_ID,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (tmp_path / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

        logger.info("Uploading %s to Kaggle as %s...", FILES, DATASET_ID)
        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(tmp_path), "--dir-mode", "zip"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Try version update if dataset already exists
            result = subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(tmp_path), "-m", "Feature refresh"],
                capture_output=True,
                text=True,
            )
        if result.returncode == 0:
            logger.info("Upload complete: %s", DATASET_ID)
        else:
            logger.error("Upload failed: %s", result.stderr)
            raise RuntimeError(result.stderr)


if __name__ == "__main__":
    upload()
