#!/usr/bin/env python
"""ISO 27001 pip-audit wrapper — forces project .venv (not host Python).

Document ID: ALICE-DEV-AUDIT-VENV
Version: 1.0.0
Date: 2026-05-09
Standards: ISO 27001 §A.14.2.5 (Secure system engineering principles)

Why this wrapper exists:
    pre-commit hooks declared with `language: system` inherit the user's host
    Python from PATH. On developer machines, the host Python may contain
    accumulated tooling (jupyter, langchain, kaggle CLI deps, etc.) entirely
    unrelated to Alice-Engine. Running `pip-audit --local` on the host Python
    surfaces CVEs in irrelevant packages, polluting the security signal.

    This wrapper resolves the project's isolated `.venv/` and runs pip-audit
    against it, so `--local` audits only what Alice-Engine actually depends on.

Behavior:
    1. Resolve `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (POSIX).
    2. Fail fast with remediation if `.venv/` is missing.
    3. Invoke `python -m pip_audit --strict --local` with the project's
       documented ignore-list (CVEs with no upstream fix or accepted risk).
    4. Propagate pip-audit's exit code.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Documented ignore-list — CVEs with no upstream fix OR accepted risk.
# Each entry MUST cite the package, the reason, and a re-check date.
IGNORED_CVES: list[tuple[str, str]] = [
    # autogluon transitive deps (ADR-011 — autogluon eliminated, code legacy)
    ("CVE-2025-69872", "autogluon transitive, no fix; ADR-011 elim"),
    ("GHSA-rf74-v2fm-23pw", "nltk 3.9.3 — no fix available"),
    ("CVE-2026-33230", "nltk 3.9.3 — no fix available"),
    ("CVE-2026-33231", "nltk 3.9.3 — no fix available"),
    ("CVE-2026-25990", "pillow 11.3 — autogluon caps <12 (legacy)"),
    ("CVE-2026-4539", "pygments 2.19.2 — 2.20.0 breaks pymdownx"),
    # mlflow CVEs without upstream fix (re-check 2026-08)
    ("CVE-2026-0545", "mlflow 3.10.1 — no fix version published; re-check 2026-08"),
    ("CVE-2026-33866", "mlflow 3.10.1 — no fix version published; re-check 2026-08"),
    # pip self-CVE without fix (re-check 2026-08)
    ("CVE-2026-3219", "pip 26.0.1 — no fix version published; re-check 2026-08"),
    # idna 3.13 — fix 3.15 available, transitive (urllib3/requests). Defer to
    # dedicated dep-update session post-Phase 4a (re-check 2026-08).
    ("CVE-2026-45409", "idna 3.13 — fix 3.15 avail, transitive; defer to dep-update post-Phase 4a"),
    # pymdown-extensions 10.21.2 — fix 10.21.3 avail, mkdocs-only.
    # Defer to dep-update session (re-check 2026-08).
    (
        "CVE-2026-46338",
        "pymdown-extensions 10.21.2 — fix 10.21.3 avail, mkdocs-only; defer dep-update",
    ),
    # starlette 0.52.1 — fix 1.0.1 requires FastAPI major bump (FastAPI 0.136.1
    # incompatible with starlette 1.x). Re-check on FastAPI 1.x release.
    (
        "PYSEC-2026-161",
        "starlette 0.52.1 — FastAPI 0.136 incompatible with starlette 1.0; re-check on FastAPI bump",
    ),
    # torch 2.9.1 — no upstream fix published yet (re-check 2026-08).
    ("PYSEC-2026-139", "torch 2.9.1 — no upstream fix published; re-check 2026-08"),
    # torch 2.9.1 — MEDIUM (CVSS 5.3) local-only crashes in torch APIs ALICE
    # never imports (transitive via TabPFN, autogluon path eliminated ADR-011).
    # Proper fix = drop torch/TabPFN with autogluon dead code (debt, dep-cleanup).
    (
        "CVE-2025-3000",
        "torch 2.9.1 — MEDIUM local DoS in torch.jit.script, no fix; not imported (TabPFN/AG-elim); re-check 2026-08",
    ),
    (
        "CVE-2025-3001",
        "torch 2.9.1 — MEDIUM local mem-corruption in torch.lstm_cell, fix 2.10.0; not imported (TabPFN/AG-elim); re-check 2026-08",
    ),
]


def find_venv_python() -> Path:
    """Resolve the project's `.venv/` Python interpreter (Windows or POSIX layout)."""
    if os.name == "nt":
        candidate = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return candidate


def main() -> int:
    """Run pip-audit against the project `.venv/`; exit with pip-audit's status."""
    venv_python = find_venv_python()
    if not venv_python.exists():
        sys.stderr.write(
            f"ERROR: project .venv missing at {venv_python}\n"
            f"Remediation:\n"
            f"  python -m venv .venv\n"
            f"  .venv/Scripts/pip install -r requirements.txt -r requirements-dev.txt  (Windows)\n"
            f"  .venv/bin/pip install -r requirements.txt -r requirements-dev.txt      (POSIX)\n"
        )
        return 2

    if not shutil.which(str(venv_python)) and not venv_python.is_file():
        sys.stderr.write(f"ERROR: {venv_python} is not executable\n")
        return 2

    cmd = [str(venv_python), "-m", "pip_audit", "--strict", "--local"]
    for cve, _reason in IGNORED_CVES:
        cmd += ["--ignore-vuln", cve]

    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
