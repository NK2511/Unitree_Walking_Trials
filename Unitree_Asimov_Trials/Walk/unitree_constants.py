"""Backward-compatible wrapper for Unitree G1 robot constants.

Single source of truth is now in `robots.unitree_g1.constants`.
"""

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from robots.unitree_g1.constants import *
