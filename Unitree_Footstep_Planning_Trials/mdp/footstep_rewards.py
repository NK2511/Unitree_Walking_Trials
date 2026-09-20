"""Reward functions for Angad footstep planning environment.

Five reward signals:

1. ``footstep_hit_reward``     — Sparse bonus when a foot lands within the
                                  target radius (primary footstep signal).

2. ``footstep_progress_reward`` — Continuous reward for pelvis proximity to the
                                   midpoint between the two upcoming targets.

3. ``clock_contact_penalty``    — Penalises wrong foot contact timing (kept for
                                   reference, superseded by foot_clock_reward).

4. ``foot_clock_reward``        — Rohan P. Singh-style reward that gives POSITIVE
                                   signal for correct stance force AND correct swing
                                   velocity, using the tan(pi/4 * clock * normed)
                                   formulation. This is the primary gait-rhythm reward.

5. ``pelvis_height_reward``     — Continuous reward for maintaining correct pelvis
                                   height. Prevents the robot from collapsing during
                                   early training when footstep-hit rewards are zero.

All functions follow the mjlab reward term convention:
  - First argument is ``env: ManagerBasedRlEnv``
  - Returns a 1-D FloatTensor of shape ``[B]``
  - Positive = reward, negative = penalty (weight is set in env_cfg.py)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Footstep Hit Reward
# ─────────────────────────────────────────────────────────────────────────────

class footstep_hit_reward:
    """Dense always-on foot proximity reward.

    Gives exp(-dist/sigma) at ALL distances (not gated to a radius).
    This provides a continuous gradient pulling the correct foot toward
    its target throughout the whole episode, solving the sparse-reward
    problem that prevented the policy from learning where to step.

    Args:
        sigma: Exponential decay distance. 0.5m = gradient fades over ~1m.
               The footstep_manager.target_radius (0.08m) is still used for
               the hit_rate logging metric.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sensor_name = cfg.params["sensor_name"]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        sigma: float = 0.15,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        robot: Entity = env.scene[asset_cfg.name]

        # World-frame foot site positions [B, 2, 3]
        site_pos = robot.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, 2, 3]
        left_foot_pos  = site_pos[:, 0, :]   # [B, 3]
        right_foot_pos = site_pos[:, 1, :]   # [B, 3]

        if not hasattr(env, "footstep_manager"):
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

        # Current target position in world frame [B, 3]
        target_pos = env.footstep_manager.get_target_pos_world()  # [B, 3]

        # Distance from the CORRECT foot to its target
        left_dist  = torch.norm(left_foot_pos  - target_pos, dim=1)  # [B]
        right_dist = torch.norm(right_foot_pos - target_pos, dim=1)  # [B]
        t1_is_left = env.footstep_manager.get_active_target_is_left()
        target_dist = torch.where(t1_is_left, left_dist, right_dist)

        in_radius = target_dist < env.footstep_manager.target_radius

        # DENSE always-on exponential — no sparse gate
        # Provides gradient at all distances, not just inside 8cm
        hit_reward = torch.exp(-target_dist / sigma)

        # Log for TensorBoard
        env.extras["log"]["Footstep/hit_rate"] = in_radius.float().mean()
        env.extras["log"]["Footstep/mean_foot_to_target_dist"] = target_dist.mean()

        return hit_reward




# ─────────────────────────────────────────────────────────────────────────────
# 2. Footstep Progress Reward
# ─────────────────────────────────────────────────────────────────────────────

def footstep_progress_reward(
    env: ManagerBasedRlEnv,
    sigma: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward pelvis proximity to the CENTERLINE MIDPOINT between t1 and t2 targets.

    The foot targets are 12 cm off to the side of the pelvis walking path.
    If we reward the pelvis to move toward the foot target directly, it will
    walk sideways zig-zagging or converge feet together under the pelvis.

    Instead we use the midpoint of t1 and t2 (i.e. the spline centerline),
    which is the correct location the pelvis should pass through:

        progress_reward = exp( -pelvis_dist_to_midpoint / sigma )

    Returns:
        Float tensor of shape [B] in range (0, 1].
    """
    robot: Entity = env.scene[asset_cfg.name]
    pelvis_pos_xy = robot.data.root_link_pos_w[:, :2]  # [B, 2]

    if not hasattr(env, "footstep_manager"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    # Centerline midpoint between t1 and t2 in world frame [B, 3]
    midpoint_world = env.footstep_manager.get_target_midpoint_world()  # [B, 3]
    midpoint_xy = midpoint_world[:, :2]                                 # [B, 2]

    dist = torch.norm(pelvis_pos_xy - midpoint_xy, dim=1)  # [B]
    reward = torch.exp(-dist / sigma)

    env.extras["log"]["Footstep/pelvis_to_midpoint_dist"] = dist.mean()
    return reward


# ─────────────────────────────────────────────────────────────────────────────
# 3. Clock-Based Contact Penalty (kept, but superseded by foot_clock_reward)
# ─────────────────────────────────────────────────────────────────────────────

class clock_contact_penalty:
    """Penalise the robot for having the wrong foot on the ground at the wrong time.

    The gait clock phase drives expectations:
      - phase [0.0 → 0.5): left foot should be in the air  (RIGHT foot stance)
      - phase [0.5 → 1.0): right foot should be in the air (LEFT  foot stance)

    Any deviation from this expectation is penalised linearly.

    Note: this is registered with a *negative* weight in env_cfg.py.
    Note: foot_clock_reward (below) is the preferred replacement — it also
          rewards correct behaviour, not only penalises wrong behaviour.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sensor_name = cfg.params["sensor_name"]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        gait_frequency: float = 1.25,
    ) -> torch.Tensor:
        if not hasattr(env, "footstep_manager"):
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

        gait_phase = env.footstep_manager.gait_phase

        # Read foot contact
        sensor: ContactSensor = env.scene[sensor_name]
        assert sensor.data.found is not None
        in_contact = sensor.data.found > 0  # [B, 2]
        left_contact  = in_contact[:, 0]    # [B]
        right_contact = in_contact[:, 1]    # [B]

        # Expected contact: left in air during 0.0-0.5, right in air during 0.5-1.0
        left_should_swing  = gait_phase < 0.5
        right_should_swing = gait_phase >= 0.5

        # Penalty: foot is on ground when it should be in the air
        left_penalty  = (left_should_swing  & left_contact).float()
        right_penalty = (right_should_swing & right_contact).float()

        return left_penalty + right_penalty  # [B] — multiplied by negative weight


# ─────────────────────────────────────────────────────────────────────────────
# 4. Foot Clock Reward (Rohan-style: +reward for CORRECT contact/velocity)
# ─────────────────────────────────────────────────────────────────────────────

class foot_clock_reward:
    """Faithfully mirrors Rohan P. Singh's Pchip-based gait clock design.

    Four-window gait cycle per period (matching Rohan's swing=0.75s, stance=0.35s):
      [0.000 - 0.341): RIGHT foot swings  — r_frc=-1, r_vel=+1, l_frc=+1, l_vel=-1
      [0.341 - 0.500): Double stance      — both frc=+1, both vel=-1
      [0.500 - 0.841): LEFT foot swings   — l_frc=-1, l_vel=+1, r_frc=+1, r_vel=-1
      [0.841 - 1.000): Double stance      — both frc=+1, both vel=-1

    Smooth cosine transitions (10% per boundary) replace Rohan's Pchip spline.
    STANDING mode (5% of episodes): both frc=+1, vel=-1 — robot just balances.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sensor_name = cfg.params["sensor_name"]
        self.robot_mass: float = cfg.params.get("robot_mass", 25.0)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        gait_frequency: float = 1.25,
        robot_mass: float = 25.0,
        max_foot_vel: float = 0.2,   # Rohan uses 0.2 m/s
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if not hasattr(env, "footstep_manager"):
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

        gait_phase = env.footstep_manager.gait_phase  # [B] in [0, 1)
        gait_mode  = env.footstep_manager.gait_mode   # [B] int: 0=WALKING, 1=STANDING

        # ── Phase windows (normalized fractions of Rohan's 2.20s full cycle) ──
        R_SWING_END = 0.75 / 2.20   # 0.3409
        DBL1_END    = 1.10 / 2.20   # 0.5000
        L_SWING_END = 1.85 / 2.20   # 0.8409
        RELAX       = 0.10           # Rohan's strict_relaxer

        def _smooth_gate(p: torch.Tensor, start: float, end: float) -> torch.Tensor:
            """Cosine-smooth gate: 0 outside [start,end], ramps 0→1→0 inside."""
            r = RELAX * (end - start)
            entry = torch.clamp((p - start) / r, 0.0, 1.0)
            exit_ = torch.clamp((end   - p) / r, 0.0, 1.0)
            return torch.minimum(entry, exit_)

        r_gate = _smooth_gate(gait_phase, 0.0,      R_SWING_END)
        l_gate = _smooth_gate(gait_phase, DBL1_END, L_SWING_END)

        # frc: +1=push (stance), -1=lift (swing). vel: +1=move (swing), -1=still (stance)
        r_frc_clock = 1.0 - 2.0 * r_gate
        r_vel_clock = 2.0 * r_gate - 1.0
        l_frc_clock = 1.0 - 2.0 * l_gate
        l_vel_clock = 2.0 * l_gate - 1.0

        # ── STANDING mode override (Rohan's WalkModes.STANDING: 5% of episodes) ─
        standing = (gait_mode == 1).float()
        r_frc_clock = standing + (1.0 - standing) * r_frc_clock
        r_vel_clock = -standing + (1.0 - standing) * r_vel_clock
        l_frc_clock = standing + (1.0 - standing) * l_frc_clock
        l_vel_clock = -standing + (1.0 - standing) * l_vel_clock

        # ── Ground reaction force score ───────────────────────────────────────
        sensor: ContactSensor = env.scene[sensor_name]
        assert sensor.data.force is not None
        left_fz  = torch.abs(sensor.data.force[:, 0, 2])
        right_fz = torch.abs(sensor.data.force[:, 1, 2])

        desired_max_fz = robot_mass * 9.81 * 0.5
        left_fz_norm  = (left_fz.clamp(max=desired_max_fz)  / desired_max_fz) * 2.0 - 1.0
        right_fz_norm = (right_fz.clamp(max=desired_max_fz) / desired_max_fz) * 2.0 - 1.0

        pi_4 = math.pi / 4.0
        lim  = pi_4 * 0.99
        left_frc_score  = torch.tan(torch.clamp(pi_4 * l_frc_clock * left_fz_norm,  -lim, lim))
        right_frc_score = torch.tan(torch.clamp(pi_4 * r_frc_clock * right_fz_norm, -lim, lim))

        # ── Foot velocity score (Rohan's max_foot_vel = 0.2 m/s) ─────────────
        robot: Entity = env.scene[asset_cfg.name]
        site_ids, _ = robot.find_sites(["left_foot_site", "right_foot_site"])

        left_site_vel  = robot.data.site_lin_vel_w[:, site_ids[0], :]
        right_site_vel = robot.data.site_lin_vel_w[:, site_ids[1], :]

        left_vel_mag  = torch.norm(left_site_vel,  dim=1)
        right_vel_mag = torch.norm(right_site_vel, dim=1)

        _max_vel = max_foot_vel
        left_vel_norm  = (left_vel_mag.clamp(max=_max_vel)  / _max_vel) * 2.0 - 1.0
        right_vel_norm = (right_vel_mag.clamp(max=_max_vel) / _max_vel) * 2.0 - 1.0

        left_vel_score  = torch.tan(torch.clamp(pi_4 * l_vel_clock * left_vel_norm,  -lim, lim))
        right_vel_score = torch.tan(torch.clamp(pi_4 * r_vel_clock * right_vel_norm, -lim, lim))

        return 0.25 * (left_frc_score + right_frc_score + left_vel_score + right_vel_score)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pelvis Height Reward
# ─────────────────────────────────────────────────────────────────────────────

def pelvis_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float = 0.85,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for maintaining the pelvis at the correct standing height.

    Mirrors Rohan's calc_height_reward. Gives the robot a continuous positive
    signal for staying upright at the correct height. Without this reward there
    is nothing to tell the policy to keep the pelvis up when footstep-hit
    rewards are zero (i.e. during all of early training).

        r = exp( -40 * (pelvis_z - target_height)^2 )

    This is 1.0 at the target and decays to ~0 within ±0.1m.

    Args:
        target_height: Desired pelvis Z height in world frame (metres). Default 0.85 m.
    """
    robot: Entity = env.scene[asset_cfg.name]
    pelvis_z = robot.data.root_link_pos_w[:, 2]   # [B]
    error = torch.abs(pelvis_z - target_height)
    return torch.exp(-40.0 * error ** 2)           # [B]


def upper_body_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for keeping the upper body vertical (minimising head-pelvis XY distance).

    Mirrors Rohan's upper_body_reward:
        r = exp( -10 * dist_xy^2 )
    """
    robot: Entity = env.scene[asset_cfg.name]

    # Try to find a torso or neck body index
    body_names = robot.body_names
    
    torso_idx = None
    for idx, name in enumerate(body_names):
        name_lower = name.lower()
        if "torso" in name_lower or "neck" in name_lower or "head" in name_lower:
            torso_idx = idx
            break
            
    if torso_idx is None:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
        
    head_pos_xy = robot.data.body_link_pos_w[:, torso_idx, :2]
    root_pos_xy = robot.data.root_link_pos_w[:, :2]
    
    dist_sq = torch.sum(torch.square(head_pos_xy - root_pos_xy), dim=1)
    return torch.exp(-10.0 * dist_sq)


def footstep_orient_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward body orientation matching the direction of travel toward the target.

    Instead of using the raw plan waypoint's stored yaw (t1_theta), which can
    jump discontinuously between spline segments on curved paths, we derive a
    smooth heading target from the vector between the pelvis and the upcoming
    target midpoint. This avoids the spinning behavior caused by discontinuous
    yaw targets while still discouraging the robot from rotating away from its
    direction of travel.

        heading_target = atan2(midpoint.y - pelvis.y, midpoint.x - pelvis.x)
        orient_cost = exp( -10 * (1 - <q_target, q_body>^2) )
    """
    if not hasattr(env, "footstep_manager"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    robot: Entity = env.scene[asset_cfg.name]
    root_quat = robot.data.root_link_quat_w  # [B, 4] (w, x, y, z)
    pelvis_xy = robot.data.root_link_pos_w[:, :2]  # [B, 2]

    midpoint_world = env.footstep_manager.get_target_midpoint_world()  # [B, 3]
    midpoint_xy = midpoint_world[:, :2]

    direction = midpoint_xy - pelvis_xy  # [B, 2]
    dist = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-4)
    direction = direction / dist  # normalize; avoids NaN heading when very close

    heading_target = torch.atan2(direction[:, 1], direction[:, 0])  # [B]

    cos_half = torch.cos(heading_target * 0.5)
    sin_half = torch.sin(heading_target * 0.5)
    target_quat = torch.stack([
        cos_half,
        torch.zeros_like(heading_target),
        torch.zeros_like(heading_target),
        sin_half,
    ], dim=1)  # [B, 4]

    inner_prod = torch.sum(root_quat * target_quat, dim=1)  # [B]
    error = 10.0 * (1.0 - inner_prod ** 2)
    return torch.exp(-error)


def pelvis_collapsed(
    env: ManagerBasedRlEnv,
    min_height: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate the episode if the pelvis height relative to feet drops below a threshold.

    Mirrors Rohan's done condition:
        terminate = (root_rel_height < 0.6)
    """
    robot: Entity = env.scene[asset_cfg.name]

    # Pelvis world-frame Z pos
    pelvis_z = robot.data.root_link_pos_w[:, 2]  # [B]

    # Foot site world-frame Z pos
    site_ids, _ = robot.find_sites(["left_foot_site", "right_foot_site"])
    left_foot_z  = robot.data.site_pos_w[:, site_ids[0], 2]   # [B]
    right_foot_z = robot.data.site_pos_w[:, site_ids[1], 2]   # [B]
    min_foot_z = torch.minimum(left_foot_z, right_foot_z)     # [B]

    relative_height = pelvis_z - min_foot_z
    return relative_height < min_height


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dynamic Kinematic Trajectory Imitation Reward
# ─────────────────────────────────────────────────────────────────────────────

class dynamic_kinematic_imitation_reward:
    """Dynamic Kinematic Trajectory Imitation Reward.

    Computes on-the-fly reference joint angles q_ref(phi) matching the exact
    kinematic arc walk trajectory for the current active footstep target,
    and rewards matching it:
        r = exp( - ||q_act - q_ref(phi)||^2 / std^2 )

    Args:
        std: Gaussian kernel width for joint angle matching (default 0.5 rad).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.std = cfg.params.get("std", 0.5)
        from .kinematic_trajectory import KinematicTrajectoryGenerator
        from .lipm_dcm_generator import LipmDcmTrajectoryGenerator
        self.kinematic_gen = KinematicTrajectoryGenerator(env.num_envs, env.device)
        self.lipm_gen = LipmDcmTrajectoryGenerator(
            num_envs=env.num_envs,
            device=env.device,
            com_height=0.831,
            step_time=1.10,
            dsp_ratio=0.20,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float = 0.5,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if not hasattr(env, "footstep_manager"):
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

        robot: Entity = env.scene[asset_cfg.name]

        # Get swing phase normalized to [0, 1) within active swing
        gait_phase = env.footstep_manager.get_swing_phase()
        t1_is_left = env.footstep_manager.get_active_target_is_left() # [B] bool
        active_foot = torch.where(
            t1_is_left,
            torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
            torch.ones(env.num_envs, device=env.device, dtype=torch.long),
        )

        # Stance & swing foot positions
        site_pos = robot.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, 2, 3]
        left_foot  = site_pos[:, 0, :]   # [B, 3]
        right_foot = site_pos[:, 1, :]   # [B, 3]

        p_stance      = torch.where(active_foot.unsqueeze(-1) == 0, right_foot, left_foot)
        p_swing_start = torch.where(active_foot.unsqueeze(-1) == 0, left_foot, right_foot)
        t1_target     = env.footstep_manager._gather_target(env.footstep_manager.t1) # [B, 4] (x, y, z, theta)
        t2_target     = env.footstep_manager._gather_target(env.footstep_manager.t2) # [B, 4]

        # Compute LIPM CoM for pelvis reference so IK pelvis matches LIPM CoM 100% (Bug 1 fix)
        p_com, _, _ = self.lipm_gen.compute_dynamic_com(
            gait_phase, p_stance, p_swing_start, t1_target, t2_target
        )

        arc_h = getattr(env.footstep_manager, "arc_height", 0.12)

        # Compute dynamic reference joints on GPU
        q_ref = self.kinematic_gen.compute_reference_joints(
            gait_phase=gait_phase,
            active_foot=active_foot,
            p_stance=p_stance,
            p_swing_start=p_swing_start,
            t1_target=t1_target,
            arc_height=arc_h,
            p_pelvis_override=p_com,
        )

        # Actual leg joint positions [B, 12] (select 12 leg joints)
        q_act = robot.data.joint_pos[:, asset_cfg.joint_ids][:, :12]  # [B, 12]

        # Gaussian error reward
        joint_err = torch.sum((q_act - q_ref) ** 2, dim=1)  # [B]
        reward = torch.exp(-joint_err / (self.std ** 2))
        return reward


def target_stagnated(
    env: ManagerBasedRlEnv,
    max_stagnant_steps: int = 120,  # 120 * 0.02s = 2.4s max time allowed per step target
) -> torch.Tensor:
    """Terminate episode if the robot fails to advance to the next target within 2.4 seconds."""
    if not hasattr(env, "footstep_manager"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return env.footstep_manager.steps_since_hit >= max_stagnant_steps


# ─────────────────────────────────────────────────────────────────────────────
# 7. Dense LIPM CoM Trajectory Tracking Reward
# ─────────────────────────────────────────────────────────────────────────────

class lipm_com_tracking_reward:
    """Dense LIPM Center of Mass (CoM) Trajectory Tracking Reward.

    Computes analytical 3D LIPM CoM trajectory p_com(t) on GPU across all
    parallel environments and rewards the robot pelvis for matching it:
        r = exp( - ||p_pelvis - p_com||^2 / sigma^2 )

    Prevents early-training torso lurching/wobbling and enforces dynamic balance.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sigma = cfg.params.get("sigma", 0.15)
        from .lipm_dcm_generator import LipmDcmTrajectoryGenerator
        self.lipm_gen = LipmDcmTrajectoryGenerator(
            num_envs=env.num_envs,
            device=env.device,
            com_height=0.831,
            step_time=1.10,
            dsp_ratio=0.20,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sigma: float = 0.15,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if not hasattr(env, "footstep_manager"):
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

        robot: Entity = env.scene[asset_cfg.name]
        pelvis_pos = robot.data.root_link_pos_w  # [B, 3]

        # Get swing phase normalized to [0, 1) within active swing
        gait_phase = env.footstep_manager.get_swing_phase()
        t1_is_left = env.footstep_manager.get_active_target_is_left()

        site_ids, _ = robot.find_sites(["left_foot_site", "right_foot_site"])
        left_foot  = robot.data.site_pos_w[:, site_ids[0], :]   # [B, 3]
        right_foot = robot.data.site_pos_w[:, site_ids[1], :]   # [B, 3]

        p_stance      = torch.where(t1_is_left.unsqueeze(-1), right_foot, left_foot)
        p_swing_start = torch.where(t1_is_left.unsqueeze(-1), left_foot, right_foot)

        t1_target = env.footstep_manager._gather_target(env.footstep_manager.t1)  # [B, 4]
        t2_target = env.footstep_manager._gather_target(env.footstep_manager.t2)  # [B, 4]

        # Compute continuous analytical LIPM CoM position on GPU
        p_com, _, _ = self.lipm_gen.compute_dynamic_com(
            gait_phase, p_stance, p_swing_start, t1_target, t2_target
        )

        dist_sq = torch.sum((pelvis_pos - p_com) ** 2, dim=1)  # [B]
        return torch.exp(-dist_sq / (sigma ** 2))


