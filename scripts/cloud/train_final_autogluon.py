"""V9 AutoGluon Benchmark — best_quality, multiclass, 204 features."""

import subprocess
import sys
from pathlib import Path

# Install AutoGluon (not pre-installed on Kaggle)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "autogluon>=1.5"],
    stdout=subprocess.DEVNULL,
)

for p in [
    Path("/kaggle/input/alice-code"),
    Path("/kaggle/input/datasets/pguillemin/alice-code"),
]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from scripts.cloud.train_autogluon_v9 import main  # noqa: E402

main()
