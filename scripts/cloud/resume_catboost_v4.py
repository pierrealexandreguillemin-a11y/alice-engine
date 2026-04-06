"""Entry point: CatBoost v4 from scratch (lr=0.005, iterations=100K, CPU).

From scratch with same lr as v3 but iterations=100K for convergence.
Snapshot (snapshot_interval=600) enables resume in session 2 if timeout.
Session 2: upload snapshot as dataset, copy to /kaggle/working/, relaunch with SAME params.
"""

import os
import sys
from pathlib import Path

os.environ["ALICE_MODEL"] = "catboost"

# Setup sys.path for Kaggle (alice-code dataset)
for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

# If snapshot dataset exists (session 2+), copy to /kaggle/working/ for CatBoost to find
snap_candidates = [
    Path("/kaggle/input/alice-catboost-v4-snapshot/catboost_snapshot"),
    Path("/kaggle/input/datasets/pguillemin/alice-catboost-v4-snapshot/catboost_snapshot"),
]
for snap in snap_candidates:
    if snap.exists():
        import shutil

        dst = Path("/kaggle/working/catboost_snapshot")
        shutil.copy2(snap, dst)
        print(f"Snapshot restored: {snap} -> {dst} ({snap.stat().st_size / 1e6:.1f} MB)")
        break
else:
    print("No snapshot found — training from scratch")

from scripts.cloud.train_kaggle import main  # noqa: E402

main()
