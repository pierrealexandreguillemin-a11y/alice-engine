"""V9 Training Final — CatBoost (alpha=0.3, oblivious, Tier 1 flags)."""

import os
import sys
from pathlib import Path

os.environ["ALICE_MODEL"] = "catboost"

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from scripts.cloud.train_kaggle import main  # noqa: E402

main()
