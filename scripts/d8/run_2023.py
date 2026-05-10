"""D8 saison 2023 thin wrapper — sets env vars before importing shared run.py.

Document ID: ALICE-D8-RUN-2023
Version: 1.0.0
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["ALICE_SAISON"] = "2023"
os.environ.setdefault("JOUEURS_PARQUET", "/kaggle/input/alice-d8-input/data/joueurs.parquet")
os.environ.setdefault("ECHIQUIERS_PARQUET", "/kaggle/input/alice-d8-input/data/echiquiers.parquet")
os.environ.setdefault("FFE_RULES_DIR", "/kaggle/input/alice-d8-input/config/ffe_rules")
os.environ.setdefault("MODEL_CACHE_DIR", "/kaggle/input/alice-d8-input/artefacts")
os.environ.setdefault("FEATURE_STORE_PATH", "/kaggle/input/alice-d8-input/data/feature_store")
os.environ.setdefault("FALLBACK_MODE", "false")

if Path("/kaggle/input").is_dir():
    for _p in Path("/kaggle/input").glob("**/scripts/d8/run.py"):
        _root = str(_p.parents[2])
        if _root not in sys.path:
            sys.path.insert(0, _root)
            break

from scripts.d8.run import main  # noqa: E402

if __name__ == "__main__":
    main()
