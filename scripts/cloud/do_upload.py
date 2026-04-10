"""Upload alice-code dataset.

Usage: python scripts/cloud/do_upload.py [notes]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cloud.upload_all_data import upload

notes = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
upload(version_notes=notes)
