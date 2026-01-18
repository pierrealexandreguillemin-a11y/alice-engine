"""Tool installation and metrics checking functions."""

from __future__ import annotations

import subprocess  # nosec B404 - subprocess for internal dev tools only
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent.parent


def _get_startupinfo() -> Any:
    """Get startupinfo for Windows to prevent console window blocking."""
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        return startupinfo
    return None


def check_installed(cmd: str) -> tuple[bool, str]:
    """Check if a tool is installed.

    Uses shell=True for cross-platform compatibility with python -m commands.
    """
    try:
        result = subprocess.run(  # noqa: S602  # nosec B602, B603, B607
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            shell=True,  # Required for "python -m" and quoted args on Windows
            startupinfo=_get_startupinfo(),
        )
        if result.returncode != 0:
            return False, "non installe"
        version = result.stdout.strip() or result.stderr.strip()
        # Extract version number
        version = version.split("\n")[0][:50] if version else "installed"
        return True, version
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "non installe"


def get_test_coverage() -> str:
    """Get current test coverage."""
    try:
        result = subprocess.run(  # noqa: S602  # nosec B602, B603, B607
            "python -m pytest --cov=app --cov=services --cov-report=term -q --tb=no",
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=120,
            shell=True,
            startupinfo=_get_startupinfo(),
        )
        for line in result.stdout.split("\n"):
            if "TOTAL" in line:
                parts = line.split()
                for part in parts:
                    if "%" in part:
                        return part
        return "N/A"
    except Exception:
        return "N/A"


def get_complexity_avg() -> str:
    """Get average complexity."""
    try:
        result = subprocess.run(  # noqa: S602  # nosec B602, B603, B607
            "python -m radon cc app services -a",
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=30,
            shell=True,
            startupinfo=_get_startupinfo(),
        )
        for line in result.stdout.split("\n"):
            if "Average complexity" in line:
                return line.split(":")[-1].strip()
        return "N/A"
    except Exception:
        return "N/A"
