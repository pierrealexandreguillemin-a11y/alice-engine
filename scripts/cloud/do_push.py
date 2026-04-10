"""Push kernel to Kaggle.

Usage: python scripts/cloud/do_push.py <metadata-suffix>
"""

import json
import shutil
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

suffix = sys.argv[1] if len(sys.argv) > 1 else "grid-search"
cloud = Path(__file__).parent

src = cloud / f"kernel-metadata-{suffix}.json"
dst = cloud / "kernel-metadata.json"
shutil.copy2(src, dst)
meta = json.loads(dst.read_text())
print(f"Pushing: {meta['id']} (code_file={meta['code_file']})")

api = KaggleApi()
api.authenticate()
api.kernels_push_cli(str(cloud), timeout=None, acc=None)
print("Done.")
