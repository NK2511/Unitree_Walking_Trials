"""Observation terms for Angad footstep planning environment.

These functions are registered as observation terms in env_cfgs.py and are
called every RL step by the mjlab ObservationManager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def footstep_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the next two footstep targets expressed in the robot's body frame,
    along with indicator flags for whether each target requires the Left (+1.0) or Right (-1.0) foot.

    Returns:
        Tensor of shape [B, 10]:
          [t1_x, t1_y, t1_z, t1_dtheta, t1_is_left,
           t2_x, t2_y, t2_z, t2_dtheta, t2_is_left]
    """
    if not hasattr(env, "footstep_manager"):
        return torch.zeros((env.num_envs, 10), device=env.device, dtype=torch.float32)
    robot: Entity = env.scene["robot"]
    root_pos_w  = robot.data.root_link_pos_w   # [B, 3]
    root_quat_w = robot.data.root_link_quat_w  # [B, 4]
    
    # [B, 8] local targets (x, y, z, theta)
    local_targets = env.footstep_manager.get_goal_steps_local(root_pos_w, root_quat_w)
    
    # Active foot indicators (+1.0 for Left, -1.0 for Right)
    t1_is_left = env.footstep_manager.get_active_target_is_left()
    t1_flag = torch.where(t1_is_left, 1.0, -1.0).unsqueeze(1)    # [B, 1]
    t2_flag = torch.where(~t1_is_left, 1.0, -1.0).unsqueeze(1)   # [B, 1] (t2 is always opposite foot of t1)
    
    # Assemble [B, 10]: [t1_x, t1_y, t1_z, t1_theta, t1_flag, t2_x, t2_y, t2_z, t2_theta, t2_flag]
    return torch.cat([
        local_targets[:, 0:4], t1_flag,
        local_targets[:, 4:8], t2_flag
    ], dim=1)



def foot_height(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Return the Z height of each foot site. Shape: [B, num_sites]."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return binary foot-contact flags. Shape: [B, 2] (left, right)."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return (sensor.data.found > 0).float()


class gait_clock:
    """Gait phase clock observation — sin/cos of a centralized fixed-frequency phase.

    Returns [cos(2π·φ), sin(2π·φ)] reading directly from env.footstep_manager.gait_phase
    so that observations and rewards are always 100% synchronized.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        pass

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        gait_frequency: float = 1.25,
    ) -> torch.Tensor:
        if not hasattr(env, "footstep_manager"):
            return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)

        phase_2pi = 2.0 * 3.14159265359 * env.footstep_manager.gait_phase
        return torch.stack([torch.cos(phase_2pi), torch.sin(phase_2pi)], dim=1)


def root_roll_pitch(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the roll and pitch angles of the robot's base body.

    Returns:
        Tensor of shape [B, 2]: [roll, pitch]
    """
    robot: Entity = env.scene["robot"]
    quat = robot.data.root_link_quat_w  # [B, 4], (w, x, y, z)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -0.99999, 0.99999))

    return torch.stack([roll, pitch], dim=1)

