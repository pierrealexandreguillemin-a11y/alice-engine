"""Data sync package for ALICE Engine.

Syncs fresh data from ffe_scrapper or HuggingFace.

Usage:
    python -m scripts.sync_data
    python -m scripts.sync_data --source hf --push
"""

from scripts.sync_data.freshness import check_freshness, check_source
from scripts.sync_data.huggingface import pull_from_hf, push_to_hf
from scripts.sync_data.symlink import update_symlink
from scripts.sync_data.types import FreshnessReport, SourceStatus, SyncConfig

__all__ = [
    # Types
    "FreshnessReport",
    "SourceStatus",
    "SyncConfig",
    # Freshness
    "check_freshness",
    "check_source",
    # Symlink
    "update_symlink",
    # HuggingFace
    "pull_from_hf",
    "push_to_hf",
]
