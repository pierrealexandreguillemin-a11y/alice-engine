"""V9 Training Final — XGBoost (alpha=0.5, depth=6, Tier 2 draw calibration)."""

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

from scripts.cloud.train_kaggle import main  # noqa: E402

main()
