"""Backward-compatible forwarder to envs.unitree_walk.mdp.velocity_command"""

import sys
from pathlib import Path
REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from envs.unitree_walk.mdp.velocity_command import *
