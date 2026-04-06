"""Kaggle Feature Engineering — ALICE Engine V8 (ISO 42001/5259/5055).

Kernel 1 of 2: generates V8 feature parquets (train/valid/test).
Output consumed by Kernel 2 (train_kaggle_v8.py) via kernel_sources.

NO GPU needed — pure CPU + RAM (30GB P100 instance).
Disables cudf to avoid groupby slowdown on feature engineering workloads.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Disable cudf — causes 10x slowdown on groupby-heavy feature engineering
os.environ["CUDF_PANDAS"] = "0"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # no GPU needed

import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))


def _setup_kaggle_imports() -> None:
    """Find alice-code dataset and add to sys.path."""
    import zipfile  # noqa: PLC0415

    candidates = [
        Path("/kaggle/input/alice-code"),
        Path("/kaggle/input/datasets/pguillemin/alice-code"),
    ]
    kaggle_input = next((c for c in candidates if c.exists()), None)
    root = Path("/kaggle/input")
    if root.exists():
        items = list(root.rglob("*"))
        logger.info("/kaggle/input/ tree (%d items): %s", len(items), [str(f) for f in items[:30]])
    logger.info("kaggle_input=%s", kaggle_input)
    if not kaggle_input:
        return
    zips = list(kaggle_input.rglob("*.zip"))
    if zips:
        wd = Path("/kaggle/working/code")
        wd.mkdir(parents=True, exist_ok=True)
        for zf in zips:
            with zipfile.ZipFile(zf) as z:
                z.extractall(wd)
        sys.path.insert(0, str(wd))
    else:
        sys.path.insert(0, str(kaggle_input))
    logger.info("sys.path += %s", sys.path[0])


def main() -> None:
    """Feature engineering only — saves parquets for training kernel."""
    logger.info("ALICE Engine — V8 Feature Engineering (cudf=DISABLED, CPU only)")
    _setup_kaggle_imports()

    # Find raw data
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    data_dir = out / "code" / "data"
    kaggle_data = Path("/kaggle/input/datasets/pguillemin/alice-code/data")
    if (kaggle_data / "echiquiers.parquet").exists():
        data_dir = kaggle_data
    elif not (data_dir / "echiquiers.parquet").exists():
        data_dir = Path("data")
    logger.info("Raw data dir: %s", data_dir)

    # Run feature engineering
    from scripts.feature_engineering import run_feature_engineering_v2  # noqa: PLC0415

    features_dir = out / "data" / "features"
    run_feature_engineering_v2(data_dir=data_dir, output_dir=features_dir, include_advanced=True)

    # Verify output
    for split in ("train", "valid", "test"):
        path = features_dir / f"{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            logger.info("%s: %d rows, %d cols", split, len(df), len(df.columns))
        else:
            logger.error("MISSING: %s", path)

    # Copy to output dir for kernel_sources consumption
    output_features = OUTPUT_DIR / "features"
    output_features.mkdir(parents=True, exist_ok=True)
    import shutil  # noqa: PLC0415

    for split in ("train", "valid", "test"):
        src = features_dir / f"{split}.parquet"
        dst = output_features / f"{split}.parquet"
        if src.exists():
            shutil.copy2(src, dst)
            logger.info("Copied %s -> %s", src, dst)

    logger.info("Feature engineering COMPLETE. Parquets in %s", output_features)


if __name__ == "__main__":
    main()
