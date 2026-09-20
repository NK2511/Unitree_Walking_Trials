"""Angad footstep planning environment configuration.

This module wires together the mjlab framework pieces for training Angad to
follow a sequence of planned footstep targets.  It inherits from the working
flat-terrain velocity-tracking config in Angad_Asimov_Trials and overrides
only the reward and observation terms that need to change.

Key differences from the velocity-tracking config:
  - Adds FootstepManager to the env and hooks it into the RL step loop.
  - Replaces velocity-tracking rewards with footstep-hit and progress rewards.
  - Removes imitation learning rewards (pure RL).
  - Adds footstep target positions (local frame) to the observation space.
  - Keeps the gait clock, foot clearance, and soft-landing rewards to ensure
    a natural, healthy walking gait.
"""

from __future__ import annotations

import sys
import os
import math
import torch

# ── Resolve the sibling Walk directory so we can import unitree_constants ──────
_WALK_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../Unitree_Asimov_Trials/Walk")
)
if _WALK_DIR not in sys.path:
    sys.path.append(_WALK_DIR)

from unitree_constants import (
    UNITREE_ACTION_SCALE,
    get_unitree_robot_cfg,
)

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mdp.footstep_manager import FootstepManager
from mdp.footstep_rewards import (
    footstep_hit_reward,
    footstep_progress_reward,
    clock_contact_penalty,
    foot_clock_reward,
    pelvis_height_reward,
    upper_body_reward,
    footstep_orient_reward,
    dynamic_kinematic_imitation_reward,
    lipm_com_tracking_reward,
    target_stagnated,
    pelvis_collapsed,
)
from mdp.observations import footstep_obs, gait_clock, root_roll_pitch


# ─────────────────────────────────────────────────────────────────────────────
# ██  REWARD WEIGHTS — edit these to tune training  ██
# ─────────────────────────────────────────────────────────────────────────────
#
#   POSITIVE REWARDS (robot gets points for doing these)
#
W_IMITATION         = 0.15      # Kinematic imitation (modest bump from 0.1; loosened kernel std=0.5)
W_LIPM_COM          = 0.5       # Match analytical 3D LIPM CoM trajectory (dynamic balance)
W_FOOTSTEP_HIT      = 2.0       # Dense foot-to-target proximity (increased — primary signal now)
W_FOOT_CLOCK        = 0.5       # Rohan phase-synchronised contact/velocity reward (gait rhythm)
W_AIR_TIME          = 0.0       # Disabled (replaced by W_FOOT_CLOCK)
W_FOOTSTEP_PROGRESS = 0.3       # RE-ENABLED — wide-basin (sigma=0.5) directional gradient
W_UPRIGHT           = 0.05      # Keep torso vertical
W_PELVIS_HEIGHT     = 0.05      # Maintain standing height (0.85 m)
W_FOOTSTEP_ORIENT   = 0.03      # RE-ENABLED at low weight, now driven by pelvis→target heading
W_POSE              = 0.0       # Disabled: all joints are leg joints
#
#   PENALTIES (robot loses points for doing these)
#
W_DOF_POS_LIMITS    = -1.0      # Joint angle limit violations
W_FOOT_CLEARANCE    = 0.0       # Disabled — conflicts with imitation arc (0.10m vs 0.12m)
W_FOOT_SLIP         = -0.3      # Stance foot sliding on ground
W_FOOT_SWING_HEIGHT = 0.0       # Disabled — conflicts with imitation arc (0.10m vs 0.12m)
W_BODY_ANG_VEL      = -0.2      # Pelvis spinning/rotating (increased from -0.08)
W_ANGULAR_MOMENTUM  = -0.03     # Whole-body angular momentum
W_ACTION_RATE       = -0.01     # Jerky action changes
W_SOFT_LANDING      = -1e-5     # Hard foot impacts
#
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Environment hook — injects FootstepManager into the env
# ─────────────────────────────────────────────────────────────────────────────

def _patch_env_with_footstep_manager(
    env: ManagerBasedRlEnv,
    plan_path: str | None = None,
    mode: str | int | None = None,
    num_spline_points: int = 4,
    step_distance: float = 0.56,
    min_segment_length: float = 3.0,
    max_segment_length: float = 5.0,
    max_angle_dev_deg: float = 40.0,
    lateral_offset: float | None = None,
) -> None:
    """Monkey-patch the FootstepManager onto the environment after construction.

    This is called from the custom runner (or from a post-init hook if mjlab
    supports it).  The manager is stored as ``env.footstep_manager`` and is
    accessible by all reward / observation terms.
    """
    if not hasattr(env, "footstep_manager"):
        env.footstep_manager = FootstepManager(  # type: ignore[attr-defined]
            env=env,
            num_steps=20,
            step_length=0.28,
            step_width=0.12,
            step_height=0.0,    # Change to 0.06 for stair climbing
            target_radius=0.15, # Larger radius: easier to hit, gives hit_rate signal early
            delay_steps=3,      # 3 × 0.02s = 0.06s debounce
            plan_path=plan_path,
            forced_mode=mode,
            num_spline_points=num_spline_points,
            step_distance=step_distance,
            min_segment_length=min_segment_length,
            max_segment_length=max_segment_length,
            max_angle_dev_deg=max_angle_dev_deg,
            lateral_offset=lateral_offset,
        )

        # Hook the manager into the env's visualizer registry for GUI rendering
        env.manager_visualizers["footstep_manager"] = env.footstep_manager

        # Hook the footstep manager's step() and reset() into the env's callbacks
        _orig_step = env.step
        _orig_reset = env.reset

        def _patched_reset(env_ids=None, **kwargs):
            ret = _orig_reset(env_ids=env_ids, **kwargs)
            if env_ids is None:
                reset_ids = torch.arange(env.num_envs, device=env.device)
            else:
                reset_ids = env_ids
            env.footstep_manager.reset(reset_ids)
            return ret
        env.reset = _patched_reset  # type: ignore[method-assign]

        if hasattr(env, "_reset_idx"):
            _orig_reset_idx = env._reset_idx
            def _patched_reset_idx(env_ids):
                ret = _orig_reset_idx(env_ids)
                env.footstep_manager.reset(env_ids)
                return ret
            env._reset_idx = _patched_reset_idx  # type: ignore[method-assign]

        def _patched_step(action):
            obs, rew, terminated, truncated, info = _orig_step(action)
            # Ensure any environments that just reset regenerate their targets right away
            reset_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
            if len(reset_ids) > 0:
                env.footstep_manager.reset(reset_ids)
            robot = env.scene["robot"]
            site_names = ("left_foot_site", "right_foot_site")
            site_ids, _ = robot.find_sites(list(site_names))
            left_foot_pos  = robot.data.site_pos_w[:, site_ids[0], :]
            right_foot_pos = robot.data.site_pos_w[:, site_ids[1], :]
            env.footstep_manager.step(left_foot_pos, right_foot_pos)
            # Auto-update 2-axis curriculum based on current RL iteration count
            current_iter = env.common_step_counter // 24
            env.footstep_manager.update_curriculum(current_iter)
            return obs, rew, terminated, truncated, info

        env.step = _patched_step  # type: ignore[method-assign]


# ─────────────────────────────────────────────────────────────────────────────
# Main Config Factory
# ─────────────────────────────────────────────────────────────────────────────

def angad_footstep_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the Angad footstep-planning environment config.

    Starts from the mjlab flat-terrain velocity-tracking base and overrides
    the reward / observation terms that need to change for footstep training.
    """
    # ── Start from the mjlab base flat-terrain config ────────────────────────
    cfg = make_velocity_env_cfg()
    cfg.scene.num_envs = 4096 if not play else 1

    # ── Robot entity ─────────────────────────────────────────────────────────
    robot_cfg = get_unitree_robot_cfg()
    _orig_spec_fn = robot_cfg.spec_fn

    def _clean_spec_fn():
        spec = _orig_spec_fn()
        try:
            floor_geom = spec.geom("floor")
            if floor_geom is not None:
                floor_geom.pos = [0, 0, -100.0]
                floor_geom.rgba = [0, 0, 0, 0]
                floor_geom.contype = 0
                floor_geom.conaffinity = 0
        except Exception:
            pass
        return spec

    robot_cfg.spec_fn = _clean_spec_fn
    cfg.scene.entities = {"robot": robot_cfg}

    # ── Foot sites ────────────────────────────────────────────────────────────
    site_names = ("left_foot_site", "right_foot_site")
    geom_names = (
        r".*left_foot_(primitive|link_geom|cap_.*).*",
        r".*right_foot_(primitive|link_geom|cap_.*).*",
    )

    # ── Sensors ───────────────────────────────────────────────────────────────
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_foot_link|right_foot_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    cfg.scene.sensors = tuple(list(cfg.scene.sensors) + [feet_ground_cfg])

    # ── Flat terrain ──────────────────────────────────────────────────────────
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Joint actions ─────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = UNITREE_ACTION_SCALE

    # ── Viewer ────────────────────────────────────────────────────────────────
    cfg.viewer.body_name = "base"

    # ── Commands: keep a zero-velocity twist so the base template doesn't error,
    #    but we don't actually use it for reward computation. ──────────────────
    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

    # ── Remove observations that reference velocity commands or are irrelevant ─
    # We remove base_lin_vel, command (twist), and projected_gravity (replaced by roll_pitch).
    for obs_name in ["base_lin_vel", "base_ang_vel", "command", "projected_gravity"]:
        cfg.observations["policy"].terms.pop(obs_name, None)
        cfg.observations["critic"].terms.pop(obs_name, None)

    # ── Add roll & pitch observation (2-D) ────────────────────────────────────
    roll_pitch_cfg = ObservationTermCfg(
        func=root_roll_pitch,
        params={},
    )
    cfg.observations["policy"].terms["root_roll_pitch"] = roll_pitch_cfg
    cfg.observations["critic"].terms["root_roll_pitch"] = roll_pitch_cfg

    # ── Add footstep target observation (8-D: t1 + t2 in local frame) ─────────
    cfg.observations["policy"].terms["footstep_targets"] = ObservationTermCfg(
        func=footstep_obs,
        params={},
    )
    cfg.observations["critic"].terms["footstep_targets"] = ObservationTermCfg(
        func=footstep_obs,
        params={},
    )

    # ── Add gait clock observation ─────────────────────────────────────────────
    gait_clock_cfg = ObservationTermCfg(
        func=gait_clock,
        params={},  # gait_frequency param removed — clock reads directly from FootstepManager
    )
    cfg.observations["policy"].terms["gait_clock"] = gait_clock_cfg
    cfg.observations["critic"].terms["gait_clock"] = gait_clock_cfg

    # ── Foot height for critic (terrain awareness) ────────────────────────────
    if "critic" in cfg.observations and "foot_height" in cfg.observations["critic"].terms:
        cfg.observations["critic"].terms["foot_height"].params[
            "asset_cfg"
        ].site_names = site_names

    # ── Reward overrides: Asimov Stability Rewards + Footstep Target Rewards ──
    # Zero out velocity-tracking rewards (we're not commanding a velocity vector)
    cfg.rewards["track_linear_velocity"].weight  = 0.0
    cfg.rewards["track_angular_velocity"].weight = 0.0

    # ── Dynamic Kinematic Trajectory Imitation ────────────────────────────────
    cfg.rewards["imitation"] = RewardTermCfg(
        func=dynamic_kinematic_imitation_reward,
        weight=W_IMITATION,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )

    # ── Dense LIPM Center of Mass (CoM) Trajectory Tracking ──────────────────
    cfg.rewards["lipm_com_tracking"] = RewardTermCfg(
        func=lipm_com_tracking_reward,
        weight=W_LIPM_COM,
        params={
            "sigma": 0.15,
            "asset_cfg": SceneEntityCfg("robot", site_names=list(site_names)),
        },
    )

    cfg.rewards["upright"].weight = W_UPRIGHT
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)

    cfg.rewards["pose"].weight = W_POSE

    cfg.rewards["body_ang_vel"].weight = W_BODY_ANG_VEL
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)

    cfg.rewards.pop("angular_momentum", None)

    cfg.rewards["air_time"].weight = W_AIR_TIME
    cfg.rewards["air_time"].params["command_name"] = None

    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].weight = W_FOOT_CLEARANCE
        cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
        cfg.rewards["foot_clearance"].params["command_name"] = None

    if "foot_swing_height" in cfg.rewards:
        cfg.rewards["foot_swing_height"].weight = W_FOOT_SWING_HEIGHT
        cfg.rewards["foot_swing_height"].params["asset_cfg"].site_names = site_names
        cfg.rewards["foot_swing_height"].params["command_name"] = None

    if "foot_slip" in cfg.rewards:
        cfg.rewards["foot_slip"].weight = W_FOOT_SLIP
        cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
        cfg.rewards["foot_slip"].params["command_name"] = None

    cfg.rewards["soft_landing"].weight = W_SOFT_LANDING
    cfg.rewards["soft_landing"].params["command_name"] = None
    cfg.rewards["action_rate_l2"].weight = W_ACTION_RATE
    if "dof_pos_limits" in cfg.rewards:
        cfg.rewards["dof_pos_limits"].weight = W_DOF_POS_LIMITS

    cfg.rewards["footstep_hit"] = RewardTermCfg(
        func=footstep_hit_reward,
        weight=W_FOOTSTEP_HIT,
        params={
            "sensor_name": feet_ground_cfg.name,
            "sigma": 0.25,   # 0.25 keeps precision near the target while giving a usable gradient further out
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
            "target_height": 0.85,
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

    # ── Disable domain randomisation items that need velocity commands ────────
    cfg.events.get("push_robot") and cfg.events.pop("push_robot", None)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names


    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["pelvis_collapsed"] = TerminationTermCfg(
        func=pelvis_collapsed,
        params={
            "min_height": 0.60,
            "asset_cfg": SceneEntityCfg("robot", site_names=list(site_names)),
        },
    )

    cfg.terminations["target_stagnated"] = TerminationTermCfg(
        func=target_stagnated,
        params={
            # Increased from 120 (2.4s) to 500 (10s). At 2.4s the robot barely
            # had time to explore stepping before the episode was killed,
            # trapping it in the 'lean forward but never step' local minimum.
            "max_stagnant_steps": 500,
        },
    )

    # ── Disable terrain curriculum ────────────────────────────────────────────
    if cfg.curriculum is not None:
        cfg.curriculum.pop("terrain_levels", None)
        cfg.curriculum.pop("command_vel", None)

    # ── Play mode tweaks ──────────────────────────────────────────────────────
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["policy"].enable_corruption = False

    return cfg
