"""Entry point: LightGBM grid search 200K subsample (CPU, 12h)."""

import os
import sys
from pathlib import Path

os.environ["ALICE_MODEL"] = "lightgbm"

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from scripts.cloud.grid_search import main  # noqa: E402

main()
