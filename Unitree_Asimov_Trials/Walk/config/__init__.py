"""Backward-compatible forwarder to envs.unitree_walk"""

import sys
from pathlib import Path
REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from envs.unitree_walk import *
from envs.unitree_walk.env_cfgs import unitree_rough_env_cfg, unitree_flat_env_cfg
from envs.unitree_walk.rl_cfg import asimov_ppo_runner_cfg
