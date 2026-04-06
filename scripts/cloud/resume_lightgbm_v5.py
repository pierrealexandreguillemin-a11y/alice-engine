"""Entry point: LightGBM v5 resume from 65K model (lr=0.005, CPU).

Resume from v4 model (65K trees, val=0.5204) — model still improving, not converged.
n_estimators=40K = ADDITIONAL iters (max 105K total). Calculated:
12h - 55min startup - 65min essential post = 10h training / 82s per 100 iters = 43.9K max.
"""

import os
import sys
from pathlib import Path

os.environ["ALICE_MODEL"] = "lightgbm"

# Setup sys.path for Kaggle (alice-code dataset)
for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

# Locate v4 model (65K trees) from dataset upload
ckpt_candidates = [
    Path("/kaggle/input/alice-lgbm-v4-model/lgbm_65k_model.txt"),
    Path("/kaggle/input/datasets/pguillemin/alice-lgbm-v4-model/lgbm_65k_model.txt"),
]
for ckpt in ckpt_candidates:
    if ckpt.exists():
        os.environ["ALICE_INIT_MODEL"] = str(ckpt)
        print(f"Resume checkpoint: {ckpt} ({ckpt.stat().st_size / 1e6:.1f} MB)")
        break
else:
    raise FileNotFoundError(f"No model found in: {[str(c) for c in ckpt_candidates]}")

from scripts.cloud.train_kaggle import main  # noqa: E402

main()
