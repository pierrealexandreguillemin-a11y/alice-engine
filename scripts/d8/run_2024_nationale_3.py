"""D8 saison 2024 Nationale 3 wrapper — probe mount + bootstrap.

Document ID: ALICE-D8-RUN-2024-nationale_3
Version: 2.0.0
"""

from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys
from pathlib import Path

os.environ["ALICE_SAISON"] = "2024"
os.environ["ALICE_DIVISION"] = "Nationale 3"
# D-2026-05-11 : bump max_matches pour buffer post-filter conformal N>=31.
os.environ.setdefault("ALICE_MAX_MATCHES", "200")


# Probe mount layout (depth-1 OR depth-4 for fresh datasets) — INLINE per
# kaggle-deployment skill 2026-05-04 chicken-and-egg trap.
def _probe(slug: str) -> Path | None:
    for c in (Path(f"/kaggle/input/{slug}"), Path(f"/kaggle/input/datasets/pguillemin/{slug}")):
        if c.is_dir():
            return c
    return None


_input = _probe("alice-d8-input")
_code = _probe("alice-d8-code")
if _input is not None:
    os.environ.setdefault("JOUEURS_PARQUET", str(_input / "data" / "joueurs.parquet"))
    os.environ.setdefault("ECHIQUIERS_PARQUET", str(_input / "data" / "echiquiers.parquet"))
    os.environ.setdefault("FFE_RULES_DIR", str(_input / "config" / "ffe_rules"))
    os.environ.setdefault("MODEL_CACHE_DIR", str(_input / "artefacts"))
    os.environ.setdefault("FEATURE_STORE_PATH", str(_input / "data" / "feature_store"))
os.environ.setdefault("FALLBACK_MODE", "false")

if _code is not None and str(_code) not in sys.path:
    sys.path.insert(0, str(_code))
    _req = _code / "scripts" / "d8" / "kaggle-requirements.txt"
    if _req.is_file():
        subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pip", "install", "--quiet", "-r", str(_req)],
            check=False,
        )

from scripts.d8.run import main  # noqa: E402

if __name__ == "__main__":
    main()
