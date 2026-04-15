"""Upload all data + code to Kaggle Dataset pguillemin/alice-code.

Everything the Kaggle kernel needs in one dataset:
- data/echiquiers.parquet (raw parsed games)
- data/joueurs.parquet (player registry)
- scripts/ (feature engineering code, training code, baselines)
- schemas/ (validation constants)

Usage: python -m scripts.cloud.upload_all_data
       python -m scripts.cloud.upload_all_data --version-notes "V8 bool fix"
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_SLUG = "pguillemin/alice-code"


def _git_version_notes() -> str:
    """Build version notes from git HEAD (short hash + subject)."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h %s"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        logger.debug("git log failed, using default version notes")
    return "manual upload"


def _compute_content_hash(tmp_path: Path) -> str:
    """SHA-256 hash of all files in the package (ISO 5259 lineage)."""
    import hashlib

    h = hashlib.sha256()
    for f in sorted(tmp_path.rglob("*")):
        if f.is_file() and f.name != "dataset-metadata.json":
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _save_upload_record(notes: str, content_hash: str) -> None:
    """Append upload record to tracking log (ISO 5259 commit↔dataset↔kernel)."""
    record_path = PROJECT_ROOT / "reports" / "kaggle_upload_log.jsonl"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "git_commit": _git_version_notes().split(" ")[0],
        "dataset_slug": DATASET_SLUG,
        "content_hash": content_hash,
        "version_notes": notes,
    }
    with open(record_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Upload record saved: commit=%s hash=%s", record["git_commit"], content_hash)


def upload(version_notes: str | None = None) -> None:
    """Package everything and upload to Kaggle (ISO 5259 tracked)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    notes = version_notes or _git_version_notes()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _package_all(tmp_path)
        content_hash = _compute_content_hash(tmp_path)
        logger.info("Content hash: %s", content_hash)

        meta = {
            "title": "alice-code",
            "id": DATASET_SLUG,
            "licenses": [{"name": "CC0-1.0"}],
            "isPrivate": True,
        }
        (tmp_path / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

        logger.info("Uploading to Kaggle as %s (notes: %s)...", DATASET_SLUG, notes)
        try:
            api.dataset_create_version(
                folder=str(tmp_path),
                version_notes=notes,
                dir_mode="zip",
            )
            logger.info("Version updated: %s", DATASET_SLUG)
        except Exception:
            api.dataset_create_new(folder=str(tmp_path), dir_mode="zip", public=False)
            logger.info("Created: %s", DATASET_SLUG)

        # Wait for Kaggle to process the new dataset version before any kernel push.
        # Previous incident: kernels pushed immediately after upload got stale V8 code.
        import time as _time

        logger.warning(
            "WAITING 120s for Kaggle dataset propagation. Do NOT push kernels until this completes."
        )
        _time.sleep(120)
        logger.info("Dataset propagation wait complete. Safe to push kernels.")

        _save_upload_record(notes, content_hash)


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
    cloud_modules = [
        "autogluon_diagnostics.py",
        "autogluon_model_card.py",
        "train_kaggle.py",
        "train_oof_stack.py",
        "train_autogluon_v9.py",
        "train_meta_learner.py",
        "optuna_kaggle.py",
        "grid_search.py",
        "grid_gaps.py",
        "grid_gaps2.py",
        "grid_tier2.py",
    ]
    for cloud_module in cloud_modules:
        src = src_scripts / "cloud" / cloud_module
        if src.exists():
            shutil.copy2(src, cloud_dir / cloud_module)
    logger.info(
        "Copied cloud modules: %s",
        [m for m in cloud_modules if (src_scripts / "cloud" / m).exists()],
    )

    # Config (hyperparameters.yaml needed by Optuna kernels)
    config_src = PROJECT_ROOT / "config" / "hyperparameters.yaml"
    if config_src.exists():
        config_dir = tmp_path / "config"
        config_dir.mkdir(exist_ok=True)
        shutil.copy2(config_src, config_dir / "hyperparameters.yaml")
        logger.info("Copied config/hyperparameters.yaml")

    # Optuna study DBs (resume across Kaggle sessions)
    optuna_dir = PROJECT_ROOT / "optuna_studies"
    if optuna_dir.exists():
        for db_file in optuna_dir.glob("optuna_*.db"):
            shutil.copy2(db_file, tmp_path / db_file.name)
            logger.info("Copied Optuna study: %s", db_file.name)

    # Schemas
    shutil.copytree(
        PROJECT_ROOT / "schemas", tmp_path / "schemas", ignore=shutil.ignore_patterns("__pycache__")
    )
    logger.info("Copied schemas/")

    total = sum(1 for _ in tmp_path.rglob("*") if _.is_file())
    logger.info("Total files: %d", total)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload alice-code dataset to Kaggle")
    parser.add_argument("--version-notes", type=str, default=None, help="Custom version notes")
    args = parser.parse_args()
    upload(version_notes=args.version_notes)
