"""Symlink management with Windows fallback (ISO 27034).

On Windows, symlink creation requires Developer Mode or admin.
Falls back to .data_source config file if symlink fails.

ISO Compliance:
- ISO/IEC 27034:2011 - Secure Coding (path validation)
- ISO/IEC 5055:2021 - Code Quality (<50 lines per function)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def update_symlink(target: Path, link: Path) -> None:
    """Create or update a directory symlink, with fallback."""
    target = target.resolve()
    if not target.is_dir():
        msg = f"Target is not a directory: {target}"
        raise ValueError(msg)

    try:
        _create_symlink(target, link)
        logger.info("Symlink: %s -> %s", link, target)
    except OSError:
        logger.warning("Symlink failed (permissions?), using .data_source fallback")
        _write_data_source_fallback(target, link)


def _create_symlink(target: Path, link: Path) -> None:
    """Create or replace a directory symlink."""
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            msg = f"Path exists and is not a symlink: {link}"
            raise ValueError(msg)

    os.symlink(target, link, target_is_directory=True)


def _write_data_source_fallback(target: Path, link: Path) -> None:
    """Write target path to .data_source config file."""
    config_file = link.parent / ".data_source"
    config_file.write_text(str(target))
    logger.info("Fallback config: %s", config_file)
