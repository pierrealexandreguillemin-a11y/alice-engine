"""V9 OOF Stack — XGBoost + LightGBM 5-fold (alpha per-model, Tier 2 draw)."""

import os
import sys
from pathlib import Path

os.environ["ALICE_OOF_STACK"] = "1"

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from scripts.cloud.train_oof_stack import main  # noqa: E402

main()
