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
