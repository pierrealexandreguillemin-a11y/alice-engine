"""Entry point: XGBoost Tier 2 grid (colsample_bynode × gamma × max_delta_step)."""

import os
import sys
from pathlib import Path

os.environ["ALICE_MODEL"] = "xgboost"

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from scripts.cloud.grid_tier2 import main  # noqa: E402

main()
