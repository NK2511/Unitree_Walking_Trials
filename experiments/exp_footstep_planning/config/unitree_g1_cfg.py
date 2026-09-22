"""Unitree G1 Footstep Planning Environment Configuration.

Self-contained workspace configuration for training and deploying footstep-planning
policies on the Unitree G1 (and H1) humanoid robot platform.

Contains:
  1. UNITREE_G1_GAINS: Pre-configured KP and KD gain tuning dictionary for Unitree motors.
  2. get_unitree_g1_robot_cfg: Robot spec builder loading local xmls/unitree_g1/scene.xml.
  3. unitree_g1_footstep_env_cfg: Environment factory function for Unitree G1 footstep RL training.
"""

from __future__ import annotations

import os
import torch
from dataclasses import dataclass
from typing import Dict

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mdp.footstep_rewards import (
    dynamic_kinematic_imitation_reward,
    foot_clock_reward,
    footstep_hit_reward,
    footstep_orient_reward,
    footstep_progress_reward,
    lipm_com_tracking_reward,
    pelvis_collapsed,
    pelvis_height_reward,
)
from mdp.observations import footstep_obs, gait_clock, root_roll_pitch

# ─────────────────────────────────────────────────────────────────────────────
# 1. UNITREE G1 GAINS & MOTOR PROPERTIES
# Tune these parameters to match your physical hardware or target simulation:
# ─────────────────────────────────────────────────────────────────────────────

UNITREE_G1_GAINS: Dict[str, Dict[str, float]] = {
    # Leg joints (Stiffness N·m/rad, Damping N·m·s/rad)
    "hip_pitch":    {"kp": 100.0, "kd": 2.5, "action_scale": 0.25},
    "hip_roll":     {"kp": 100.0, "kd": 2.5, "action_scale": 0.25},
    "hip_yaw":      {"kp": 100.0, "kd": 2.5, "action_scale": 0.25},
    "knee":         {"kp": 150.0, "kd": 4.0, "action_scale": 0.25},
    "ankle_pitch":  {"kp": 40.0,  "kd": 1.0, "action_scale": 0.25},
    "ankle_roll":   {"kp": 40.0,  "kd": 1.0, "action_scale": 0.25},
    # Waist & Upper Body (held neutral by default)
    "waist":        {"kp": 200.0, "kd": 5.0, "action_scale": 0.0},
    "arm":          {"kp": 40.0,  "kd": 1.0, "action_scale": 0.0},
}

# Unitree G1 nominal standing joint positions (12 leg joints + waist/arms)
UNITREE_G1_DEFAULT_JOINTS = {
    "left_hip_pitch_joint":    -0.20,
    "left_hip_roll_joint":      0.00,
    "left_hip_yaw_joint":       0.00,
    "left_knee_joint":          0.42,
    "left_ankle_pitch_joint":  -0.22,
    "left_ankle_roll_joint":    0.00,
    "right_hip_pitch_joint":   -0.20,
    "right_hip_roll_joint":     0.00,
    "right_hip_yaw_joint":      0.00,
    "right_knee_joint":         0.42,
    "right_ankle_pitch_joint": -0.22,
    "right_ankle_roll_joint":   0.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. REWARD WEIGHTS FOR UNITREE G1
# ─────────────────────────────────────────────────────────────────────────────
W_FOOTSTEP_HIT      = 2.0       # Primary footstep landing reward
W_FOOT_CLOCK        = 0.5       # Contact force / swing velocity rhythm
W_LIPM_COM          = 0.5       # Analytical CoM trajectory tracking
W_FOOTSTEP_PROGRESS = 0.3       # Centerline progress toward target midpoint
W_IMITATION         = 0.15      # Kinematic leg arc imitation (loosened std=0.5)
W_UPRIGHT           = 0.05      # Torso verticality
W_PELVIS_HEIGHT     = 0.05      # Standing height (0.75m for G1)
W_FOOTSTEP_ORIENT   = 0.03      # Heading alignment to target midpoint
W_BODY_ANG_VEL      = -0.20     # Rotation/spinning penalty
W_FOOT_SLIP         = -0.30     # Foot sliding penalty
W_DOF_POS_LIMITS    = -1.00     # Joint limit violation penalty
W_ACTION_RATE       = -0.01     # Action jitter penalty
W_SOFT_LANDING      = -1e-5     # Foot impact penalty


def get_unitree_g1_xml_path() -> str:
    """Return absolute path to local unitree_g1 scene XML."""
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(workspace_root, "xmls", "unitree_g1", "scene.xml")
    if os.path.exists(xml_path):
        return xml_path
    raise FileNotFoundError(f"Unitree G1 XML not found at {xml_path}")


def unitree_g1_footstep_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the Unitree G1 footstep-planning environment configuration."""
    cfg = make_velocity_env_cfg()
    cfg.scene.num_envs = 4096 if not play else 1

    # Foot site names for Unitree G1
    site_names = ("left_foot", "right_foot")

    # Configure contact sensor for G1 feet
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    cfg.scene.sensors = tuple(list(cfg.scene.sensors) + [feet_ground_cfg])

    # Flat terrain plane
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Joint action scaling
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 0.25

    # Viewer target body
    cfg.viewer.body_name = "pelvis"

    # Zero velocity command placeholder
    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

    # Filter base velocity / command observations
    for obs_name in ["base_lin_vel", "base_ang_vel", "command", "projected_gravity"]:
        cfg.observations["policy"].terms.pop(obs_name, None)
        cfg.observations["critic"].terms.pop(obs_name, None)

    # Observation terms
    roll_pitch_cfg = ObservationTermCfg(func=root_roll_pitch, params={})
    cfg.observations["policy"].terms["root_roll_pitch"] = roll_pitch_cfg
    cfg.observations["critic"].terms["root_roll_pitch"] = roll_pitch_cfg

    footstep_targets_cfg = ObservationTermCfg(func=footstep_obs, params={})
    cfg.observations["policy"].terms["footstep_targets"] = footstep_targets_cfg
    cfg.observations["critic"].terms["footstep_targets"] = footstep_targets_cfg

    gait_clock_cfg = ObservationTermCfg(func=gait_clock, params={})
    cfg.observations["policy"].terms["gait_clock"] = gait_clock_cfg
    cfg.observations["critic"].terms["gait_clock"] = gait_clock_cfg

    # Zero out unneeded velocity rewards
    cfg.rewards["track_linear_velocity"].weight  = 0.0
    cfg.rewards["track_angular_velocity"].weight = 0.0

    # Configure active footstep rewards
    cfg.rewards["imitation"] = RewardTermCfg(
        func=dynamic_kinematic_imitation_reward,
        weight=W_IMITATION,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )

    cfg.rewards["lipm_com_tracking"] = RewardTermCfg(
        func=lipm_com_tracking_reward,
        weight=W_LIPM_COM,
        params={
            "sigma": 0.15,
            "asset_cfg": SceneEntityCfg("robot", site_names=list(site_names)),
        },
    )

    cfg.rewards["upright"].weight = W_UPRIGHT
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis",)

    cfg.rewards["body_ang_vel"].weight = W_BODY_ANG_VEL
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)

    cfg.rewards["footstep_hit"] = RewardTermCfg(
        func=footstep_hit_reward,
        weight=W_FOOTSTEP_HIT,
        params={
            "sensor_name": feet_ground_cfg.name,
            "sigma": 0.25,
            "asset_cfg": SceneEntityCfg("robot", site_names=list(site_names)),
        },
    )

    cfg.rewards["footstep_progress"] = RewardTermCfg(
        func=footstep_progress_reward,
        weight=W_FOOTSTEP_PROGRESS,
        params={
            "sigma": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    cfg.rewards["pelvis_height"] = RewardTermCfg(
        func=pelvis_height_reward,
        weight=W_PELVIS_HEIGHT,
        params={
            "target_height": 0.75,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    cfg.rewards["footstep_orient"] = RewardTermCfg(
        func=footstep_orient_reward,
        weight=W_FOOTSTEP_ORIENT,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    cfg.rewards["foot_clock"] = RewardTermCfg(
        func=foot_clock_reward,
        weight=W_FOOT_CLOCK,
        params={
            "sensor_name": feet_ground_cfg.name,
            "asset_cfg": SceneEntityCfg("robot", site_names=list(site_names)),
        },
    )

    # Terminations
    cfg.terminations["pelvis_collapsed"] = TerminationTermCfg(
        func=pelvis_collapsed,
        params={
            "min_height": 0.50,
            "asset_cfg": SceneEntityCfg("robot", site_names=list(site_names)),
        },
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["policy"].enable_corruption = False

    return cfg
