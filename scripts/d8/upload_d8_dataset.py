"""Upload alice-d8-input dataset to Kaggle (one-time setup + version bumps).

Stages : data parquets + MLP champion artefacts + FFE rules + alice-d8-code
into a single Kaggle dataset for the 4 saison kernels and aggregator.

Usage :
    python -m scripts.d8.upload_d8_dataset           # create or version bump
    python -m scripts.d8.upload_d8_dataset --check   # dry-run + validation only

Document ID: ALICE-D8-UPLOAD
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "alice-d8-input"
REPO = Path(__file__).resolve().parent.parent.parent

# (source, staged_subpath) — fail-fast if source missing.
ARTEFACT_SOURCES: tuple[tuple[Path, str], ...] = (
    (REPO / "data" / "joueurs.parquet", "data/joueurs.parquet"),
    (REPO / "data" / "echiquiers.parquet", "data/echiquiers.parquet"),
    (REPO / "models" / "cache" / "mlp_meta_learner.joblib", "artefacts/mlp_meta_learner.joblib"),
    (REPO / "models" / "cache" / "temperature_T.joblib", "artefacts/temp_scaler.joblib"),
    (REPO / "config" / "ffe_rules" / "a02.json", "config/ffe_rules/a02.json"),
)


def _validate_sources() -> None:
    """ISO 5259 fail-fast if any input file missing."""
    missing = [str(src) for src, _ in ARTEFACT_SOURCES if not src.exists()]
    if missing:
        msg = f"Missing input files for upload: {missing}"
        raise FileNotFoundError(msg)


def _stage_files(staging: Path) -> None:
    """Copy ARTEFACT_SOURCES into staging dir per Kaggle dataset layout."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    for src, dst_subpath in ARTEFACT_SOURCES:
        dst = staging / dst_subpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def _write_dataset_metadata(staging: Path) -> None:
    """Write dataset-metadata.json per Kaggle CLI requirements."""
    metadata = {
        "title": "ALICE D8 Fairness/Robustness Input",
        "id": f"pierrax/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _kaggle_create_or_version(staging: Path) -> None:
    """Try `kaggle datasets create`; if 'already exists' fall back to `version`."""
    create = subprocess.run(  # noqa: S603 - args are static literals
        ["kaggle", "datasets", "create", "-p", str(staging), "--dir-mode", "tar"],
        capture_output=True,
        text=True,
        check=False,
    )
    if create.returncode == 0:
        return
    # Most likely cause of non-zero on second push: dataset already exists.
    subprocess.run(  # noqa: S603
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(staging),
            "-m",
            "D8 input refresh",
            "--dir-mode",
            "tar",
        ],
        check=True,
    )


def main() -> int:
    """Entry point — stage + (optionally) push to Kaggle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate sources only")
    args = parser.parse_args()

    _validate_sources()
    if args.check:
        sys.stdout.write(f"OK: {len(ARTEFACT_SOURCES)} source files present.\n")
        return 0

    staging = REPO / "build" / "kaggle" / DATASET_SLUG
    _stage_files(staging)
    _write_dataset_metadata(staging)
    _kaggle_create_or_version(staging)
    sys.stdout.write(f"Dataset {DATASET_SLUG} uploaded from {staging}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
