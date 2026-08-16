"""Legacy average fusion for BAD/GAD Grad-CAM++.

This keeps the previous element-wise mean behavior while the main script
defaults to maximum fusion. All arguments are forwarded to gradcampp_average.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = SCRIPT_DIR / "gradcampp_average.py"


if "--combine" not in sys.argv[1:]:
    sys.argv[1:1] = ["--combine", "average"]

runpy.run_path(str(TARGET), run_name="__main__")
