"""V9 OOF CatBoost — folds 2-3 (alpha=0.3, depth=5, rsm=0.7)."""

import os
import sys
from pathlib import Path

os.environ["ALICE_MODEL"] = "catboost"
os.environ["ALICE_FOLDS"] = "2,3"

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from scripts.cloud.train_oof_stack import main  # noqa: E402

main()
