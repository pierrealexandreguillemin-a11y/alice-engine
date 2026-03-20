"""Entry point for: python -m scripts.sync_data.

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

_DEFAULT_SOURCE = Path(
    os.environ.get(
        "FFE_SCRAPPER_DATA_DIR",
        str(Path(__file__).resolve().parents[2] / ".." / "ffe_scrapper" / "data"),
    )
)


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
