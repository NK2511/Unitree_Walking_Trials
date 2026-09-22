from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_squared / std**2)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize self-collisions.

  Returns the number of self-collisions detected by the specified contact sensor.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class alternating_feet_contact:
  """Reward alternating foot contacts to encourage walking gait.

  Tracks a gait phase (0 to 1) and rewards:
  - Left foot in air when phase is 0.0-0.5
  - Right foot in air when phase is 0.5-1.0
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.gait_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    self.gait_frequency = cfg.params.get("gait_frequency", 1.5)  # Hz
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
    gait_frequency: float = 1.5,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()

    # --- OLD METHOD (Fixed Frequency) ---
    # self.gait_phase = torch.fmod(self.gait_phase + self.step_dt * gait_frequency, 1.0)

    # --- NEW METHOD (Sub-linear scaling) ---
    speed_ratio = (total_command / 0.5).clamp(min=0.5, max=3.0)
    phase_delta = self.step_dt * gait_frequency * torch.sqrt(speed_ratio)
    self.gait_phase = torch.fmod(self.gait_phase + phase_delta, 1.0)

    # Check foot contact (assume 2 feet: left=0, right=1)
    assert contact_sensor.data.found is not None
    in_contact = contact_sensor.data.found > 0  # [B, 2]
    left_contact = in_contact[:, 0]
    right_contact = in_contact[:, 1]

    # Desired: left swing (not contact) when phase 0.0-0.5, right swing when 0.5-1.0
    left_should_swing = self.gait_phase < 0.5
    right_should_swing = self.gait_phase >= 0.5

    # Reward when foot is in air at the right time
    left_reward = (left_should_swing & ~left_contact).float()
    right_reward = (right_should_swing & ~right_contact).float()
    reward = left_reward + right_reward

    reward = reward * active

    # Reset phase on episode reset
    reset_ids = env.termination_manager.terminated.nonzero(as_tuple=False).flatten()
    if len(reset_ids) > 0:
      self.gait_phase[reset_ids] = torch.rand(len(reset_ids), device=env.device)

    return reward


class imitation_joint_pos:
  """Reward for tracking reference joint positions from imitation data.

  Loads cyclic walking trajectory from CSV and rewards matching the reference
  pose based on gait phase computed from velocity command.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    import numpy as np
    import os

    self.step_dt = env.step_dt
    self.gait_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    # Load imitation data
    data_path = cfg.params.get("data_path", None)
    if data_path is None or not os.path.exists(data_path):
      print(f"[WARNING] Imitation data not found at {data_path}, reward disabled")
      self.enabled = False
      self.ref_joint_pos = None
      return

    self.enabled = True
    # Load active simulation joint names
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset = env.scene[asset_cfg.name]
    if asset_cfg.joint_names is not None:
      _, sim_joint_names = asset.find_joints(asset_cfg.joint_names)
    else:
      sim_joint_names = list(asset.joint_names)

    # Read CSV header to get joint names
    with open(data_path, "r") as f:
      header = f.readline().strip().split(",")
    traj_joint_names = [h for h in header if h not in ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz"]]
    self.traj_cols = [header.index(name) for name in traj_joint_names]

    # Clean function to handle mismatches
    def clean_name(name):
      return name.replace("knee_pitch", "knee").replace("knee", "knee_pitch")

    # Build mapping indices
    self.sim_indices = []
    self.traj_indices = []
    for sim_idx, sim_name in enumerate(sim_joint_names):
      c_sim = clean_name(sim_name)
      for traj_idx, traj_name in enumerate(traj_joint_names):
        c_traj = clean_name(traj_name)
        if c_sim == c_traj:
          self.sim_indices.append(sim_idx)
          self.traj_indices.append(traj_idx)
          break

    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    ref_joint_pos = data[:, self.traj_cols]
    self.ref_joint_pos = torch.tensor(ref_joint_pos, device=env.device, dtype=torch.float32)
    self.num_frames = self.ref_joint_pos.shape[0]
    self.gait_frequency = cfg.params.get("gait_frequency", 1.25)  # Hz
    print(f"[INFO] Loaded imitation data: {self.num_frames} frames at {self.gait_frequency} Hz")
    print(f"[INFO] Dynamically mapped {len(self.sim_indices)} matching joints between simulation and reference CSV.")



  def __call__(
    self,
    env: ManagerBasedRlEnv,
    data_path: str,
    gait_frequency: float,
    std: float,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del data_path  # Used in __init__

    if not self.enabled:
      return torch.zeros(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    # Update gait phase based on velocity
    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed
    active = (total_speed > command_threshold).float()

    # --- OLD METHOD (Linear Scaling) ---
    phase_delta = self.step_dt * gait_frequency * (total_speed / 0.5).clamp(min=0.5, max=2.0)
    self.gait_phase = torch.fmod(self.gait_phase + phase_delta, 1.0)


    # Get frame index from phase
    frame_idx = (self.gait_phase * self.num_frames).long() % self.num_frames

    # Get reference joint positions for matching joints
    ref_pos = self.ref_joint_pos[frame_idx][:, self.traj_indices]

    # Get current joint positions for matching joints
    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids][:, self.sim_indices]

    # Compute error 
    error_sq = torch.sum(torch.square(current_pos - ref_pos), dim=1)

    # --- OLD METHOD (Fixed Strictness) ---
    # reward = torch.exp(-error_sq / (std ** 2))

    # --- NEW METHOD (Dynamic Strictness) ---
    # Relax standard deviation to allow agent to deviate from CSV for longer strides
    dynamic_std = std * (total_speed / 0.5).clamp(min=1.0, max=3.0)
    reward = torch.exp(-error_sq / (dynamic_std ** 2))

    # Only apply when moving
    reward = reward * active

    # Reset phase on episode reset
    reset_ids = env.termination_manager.terminated.nonzero(as_tuple=False).flatten()
    if len(reset_ids) > 0:
      self.gait_phase[reset_ids] = torch.rand(len(reset_ids), device=env.device)

    return reward



class alternating_feet_contact_speed_adaptive:
  """Reward alternating foot contacts with speed-adaptive sub-linear frequency scaling."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.gait_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    self.gait_frequency = cfg.params.get("gait_frequency", 1.5)  # Hz
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
    gait_frequency: float = 1.5,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()

    # --- NEW METHOD (Sub-linear scaling) ---
    speed_ratio = (total_command / 0.5).clamp(min=0.5, max=8.0)
    phase_delta = self.step_dt * gait_frequency * torch.sqrt(speed_ratio)
    self.gait_phase = torch.fmod(self.gait_phase + phase_delta, 1.0)

    assert contact_sensor.data.found is not None
    in_contact = contact_sensor.data.found > 0  # [B, 2]
    left_contact = in_contact[:, 0]
    right_contact = in_contact[:, 1]

    left_should_swing = self.gait_phase < 0.5
    right_should_swing = self.gait_phase >= 0.5

    left_reward = (left_should_swing & ~left_contact).float()
    right_reward = (right_should_swing & ~right_contact).float()
    reward = left_reward + right_reward

    reward = reward * active

    reset_ids = env.termination_manager.terminated.nonzero(as_tuple=False).flatten()
    if len(reset_ids) > 0:
      self.gait_phase[reset_ids] = torch.rand(len(reset_ids), device=env.device)

    return reward


class imitation_joint_pos_speed_adaptive:
  """Imitation tracking with speed-adaptive amplitude trajectory scaling and strictness."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    import numpy as np
    import os

    self.step_dt = env.step_dt
    self.gait_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    data_path = cfg.params.get("data_path", None)
    if data_path is None or not os.path.exists(data_path):
      print(f"[WARNING] Imitation data not found at {data_path}, reward disabled")
      self.enabled = False
      self.ref_joint_pos = None
      return

    self.enabled = True
    # Load active simulation joint names
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset = env.scene[asset_cfg.name]
    if asset_cfg.joint_names is not None:
      _, sim_joint_names = asset.find_joints(asset_cfg.joint_names)
    else:
      sim_joint_names = list(asset.joint_names)

    # Read CSV header to get joint names
    with open(data_path, "r") as f:
      header = f.readline().strip().split(",")
    traj_joint_names = [h for h in header if h not in ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz"]]
    self.traj_cols = [header.index(name) for name in traj_joint_names]

    # Clean function to handle mismatches
    def clean_name(name):
      return name.replace("knee_pitch", "knee").replace("knee", "knee_pitch")

    # Build mapping indices
    self.sim_indices = []
    self.traj_indices = []
    for sim_idx, sim_name in enumerate(sim_joint_names):
      c_sim = clean_name(sim_name)
      for traj_idx, traj_name in enumerate(traj_joint_names):
        c_traj = clean_name(traj_name)
        if c_sim == c_traj:
          self.sim_indices.append(sim_idx)
          self.traj_indices.append(traj_idx)
          break

    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    ref_joint_pos = data[:, self.traj_cols]
    self.ref_joint_pos = torch.tensor(ref_joint_pos, device=env.device, dtype=torch.float32)
    self.num_frames = self.ref_joint_pos.shape[0]
    self.gait_frequency = cfg.params.get("gait_frequency", 1.25)
    print(f"[INFO] Loaded imitation data: {self.num_frames} frames at {self.gait_frequency} Hz")
    print(f"[INFO] Dynamically mapped {len(self.sim_indices)} matching joints between simulation and reference CSV.")

    # Determine the neutral pose for amplitude scaling logic
    self.ref_mean_pos = self.ref_joint_pos.mean(dim=0)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    data_path: str,
    gait_frequency: float,
    std: float,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if not self.enabled:
      return torch.zeros(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed
    active = (total_speed > command_threshold).float()

    # --- NEW METHOD (Sub-linear scaling) ---
    speed_ratio = (total_speed / 0.5).clamp(min=0.5, max=8.0)
    phase_delta = self.step_dt * gait_frequency * torch.sqrt(speed_ratio)
    self.gait_phase = torch.fmod(self.gait_phase + phase_delta, 1.0)

    frame_idx = (self.gait_phase * self.num_frames).long() % self.num_frames
    base_ref_pos = self.ref_joint_pos[frame_idx]  # [B, NumTrajJoints]

    # --- TRAJECTORY AMPLITUDE SCALING ---
    # Dynamically stretch the joint target swings using sub-linear (sqrt) scaling to keep v = f * L balanced
    stride_scale = torch.sqrt(speed_ratio).clamp(min=1.0, max=2.5).unsqueeze(1)
    ref_pos_all = self.ref_mean_pos + (base_ref_pos - self.ref_mean_pos) * stride_scale

    # Extract matching joint subsets
    ref_pos = ref_pos_all[:, self.traj_indices]
    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids][:, self.sim_indices]

    # Compute error 
    error_sq = torch.sum(torch.square(current_pos - ref_pos), dim=1)

    # --- DYNAMIC STRICTNESS ---
    # Relax standard deviation to allow agent to naturally adapt its long stride
    dynamic_std = std * (total_speed / 0.5).clamp(min=1.0, max=8.0)
    reward = torch.exp(-error_sq / (dynamic_std ** 2))

    reward = reward * active

    reset_ids = env.termination_manager.terminated.nonzero(as_tuple=False).flatten()
    if len(reset_ids) > 0:
      self.gait_phase[reset_ids] = torch.rand(len(reset_ids), device=env.device)

    return reward


class variable_posture:
  """Penalize deviation from default pose, with tighter constraints when standing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))

class imitation_dynamic_gait:
  """Reward for tracking dynamically scaled joint positions based on commanded velocity.
  
  Loads an npz of precomputed anchor trajectories and performs 2D bilinear interpolation
  over target velocity and gait phase.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    import numpy as np
    import os

    self.step_dt = env.step_dt
    self.gait_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    npz_path = cfg.params.get("dynamic_npz_path", None)
    if npz_path is None or not os.path.exists(npz_path):
      print(f"[WARNING] Dynamic gait NPZ not found at {npz_path}, reward disabled")
      self.enabled = False
      return

    self.enabled = True
    data = np.load(npz_path)
    self.anchor_speeds = torch.tensor(data["speeds"], device=env.device, dtype=torch.float32)
    
    # Load trajectories: [10, 120, num_joints]
    trajectories = torch.tensor(data["trajectories"], device=env.device, dtype=torch.float32)
    
    # Load active simulation joint names
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset = env.scene[asset_cfg.name]
    if asset_cfg.joint_names is not None:
      _, sim_joint_names = asset.find_joints(asset_cfg.joint_names)
    else:
      sim_joint_names = list(asset.joint_names)

    # Load trajectory joint names
    if "joint_names" in data:
      traj_joint_names = list(data["joint_names"])
    else:
      # Legacy fallback to old mappers format
      traj_joint_names = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
                          "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll"]

    # Clean function to handle mismatches
    def clean_name(name):
      return name.replace("knee_pitch", "knee").replace("knee", "knee_pitch")

    # Build mapping indices
    self.sim_indices = []
    self.traj_indices = []
    for sim_idx, sim_name in enumerate(sim_joint_names):
      c_sim = clean_name(sim_name)
      for traj_idx, traj_name in enumerate(traj_joint_names):
        c_traj = clean_name(traj_name)
        if c_sim == c_traj:
          self.sim_indices.append(sim_idx)
          self.traj_indices.append(traj_idx)
          break

    self.anchor_trajectories = trajectories
    

    
    self.num_frames = self.anchor_trajectories.shape[1]
    
    # Calculate pre-computed actual gait frequencies for each anchor speed
    # F = v / L(v)
    v_base = 0.18
    L_base = 0.32
    L_anchors = torch.zeros_like(self.anchor_speeds)
    for i, v in enumerate(self.anchor_speeds):
      t = max(0.0, min(1.0, (v.item() - v_base) / (2.5 - v_base)))
      L = L_base + (0.58 - L_base) * t
      L_anchors[i] = L
    self.anchor_frequencies = self.anchor_speeds / L_anchors
    
    print(f"[INFO] Loaded dynamic gait anchors: {len(self.anchor_speeds)} speeds, {self.num_frames} frames")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
  ) -> torch.Tensor:

    if not self.enabled:
      return torch.zeros(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed
    active = (total_speed > command_threshold).float()

    # Interpolate speed
    v_clamped = total_speed.clamp(min=self.anchor_speeds[0].item(), max=self.anchor_speeds[-1].item())
    
    # Find indices for interpolation
    v_min = self.anchor_speeds[0].item()
    v_max = self.anchor_speeds[-1].item()
    num_anchors = len(self.anchor_speeds)
    
    idx_float = (v_clamped - v_min) / (v_max - v_min) * (num_anchors - 1)
    idx_low = idx_float.floor().long()
    idx_high = (idx_low + 1).clamp(max=num_anchors - 1)
    alpha = (idx_float - idx_low.float()).unsqueeze(1) # [B, 1]
    
    # Interpolate frequencies to find current F_RL
    freq_low = self.anchor_frequencies[idx_low]
    freq_high = self.anchor_frequencies[idx_high]
    current_freq = freq_low + (freq_high - freq_low) * alpha.squeeze(1)
    
    # Advance phase using exact frequency
    phase_delta = self.step_dt * current_freq
    self.gait_phase = torch.fmod(self.gait_phase + phase_delta, 1.0)
    
    # Get frame indices
    frame_idx_float = self.gait_phase * self.num_frames
    frame_idx_low = frame_idx_float.floor().long() % self.num_frames
    frame_idx_high = (frame_idx_low + 1) % self.num_frames
    frame_alpha = (frame_idx_float - frame_idx_float.floor()).unsqueeze(1) # [B, 1]

    # Bilinear interpolation
    traj_vlow_flow = self.anchor_trajectories[idx_low, frame_idx_low]
    traj_vlow_fhigh = self.anchor_trajectories[idx_low, frame_idx_high]
    traj_vhigh_flow = self.anchor_trajectories[idx_high, frame_idx_low]
    traj_vhigh_fhigh = self.anchor_trajectories[idx_high, frame_idx_high]
    
    ref_pos_vlow = traj_vlow_flow + (traj_vlow_fhigh - traj_vlow_flow) * frame_alpha
    ref_pos_vhigh = traj_vhigh_flow + (traj_vhigh_fhigh - traj_vhigh_flow) * frame_alpha
    ref_pos_all = ref_pos_vlow + (ref_pos_vhigh - ref_pos_vlow) * alpha
    
    # Extract matching joint subsets
    ref_pos = ref_pos_all[:, self.traj_indices]
    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids][:, self.sim_indices]
    error_sq = torch.sum(torch.square(current_pos - ref_pos), dim=1)

    dynamic_std = std * (total_speed / 0.5).clamp(min=1.0, max=3.0)
    # Relax standard deviation to allow agent to naturally adapt its long stride
    dynamic_std = std * (total_speed / 0.5).clamp(min=1.0, max=8.0)
    reward = torch.exp(-error_sq / (dynamic_std ** 2))

    reward = reward * active

    reset_ids = env.termination_manager.terminated.nonzero(as_tuple=False).flatten()
    if len(reset_ids) > 0:
      self.gait_phase[reset_ids] = torch.rand(len(reset_ids), device=env.device)

    return reward


class variable_posture:
  """Penalize deviation from default pose, with tighter constraints when standing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))

class imitation_dynamic_gait:
  """Reward for tracking dynamically scaled joint positions based on commanded velocity.
  
  Loads an npz of precomputed anchor trajectories and performs 2D bilinear interpolation
  over target velocity and gait phase.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    import numpy as np
    import os

    self.step_dt = env.step_dt
    self.gait_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    npz_path = cfg.params.get("dynamic_npz_path", None)
    if npz_path is None or not os.path.exists(npz_path):
      print(f"[WARNING] Dynamic gait NPZ not found at {npz_path}, reward disabled")
      self.enabled = False
      return

    self.enabled = True
    data = np.load(npz_path)
    self.anchor_speeds = torch.tensor(data["speeds"], device=env.device, dtype=torch.float32)
    
    # Load trajectories: [10, 120, num_joints]
    trajectories = torch.tensor(data["trajectories"], device=env.device, dtype=torch.float32)
    
    # Load active simulation joint names
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset = env.scene[asset_cfg.name]
    if asset_cfg.joint_names is not None:
      _, sim_joint_names = asset.find_joints(asset_cfg.joint_names)
    else:
      sim_joint_names = list(asset.joint_names)

    # Load trajectory joint names
    if "joint_names" in data:
      traj_joint_names = list(data["joint_names"])
    else:
      # Legacy fallback to old mappers format
      traj_joint_names = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
                          "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll"]

    # Clean function to handle mismatches
    def clean_name(name):
      return name.replace("knee_pitch", "knee").replace("knee", "knee_pitch")

    # Build mapping indices
    self.sim_indices = []
    self.traj_indices = []
    for sim_idx, sim_name in enumerate(sim_joint_names):
      c_sim = clean_name(sim_name)
      for traj_idx, traj_name in enumerate(traj_joint_names):
        c_traj = clean_name(traj_name)
        if c_sim == c_traj:
          self.sim_indices.append(sim_idx)
          self.traj_indices.append(traj_idx)
          break

    self.anchor_trajectories = trajectories
    

    
    self.num_frames = self.anchor_trajectories.shape[1]
    
    # Use frequencies directly from NPZ if present, fallback to old formula if not
    if "frequencies" in data:
      self.anchor_frequencies = torch.tensor(data["frequencies"], device=env.device, dtype=torch.float32)
      print(f"[INFO] Loaded precomputed frequencies from NPZ: {self.anchor_frequencies.tolist()}")
    else:
      v_base = 0.18
      L_base = 0.32
      L_anchors = torch.zeros_like(self.anchor_speeds)
      for i, v in enumerate(self.anchor_speeds):
        t = max(0.0, min(1.0, (v.item() - v_base) / (2.5 - v_base)))
        L = L_base + (0.58 - L_base) * t
        L_anchors[i] = L
      self.anchor_frequencies = self.anchor_speeds / L_anchors
      print(f"[INFO] No frequencies found in NPZ, using fallback linear frequency mapping")
    
    print(f"[INFO] Loaded dynamic gait anchors: {len(self.anchor_speeds)} speeds, {self.num_frames} frames")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
  ) -> torch.Tensor:

    if not self.enabled:
      return torch.zeros(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed
    active = (total_speed > command_threshold).float()

    # Interpolate speed
    v_clamped = total_speed.clamp(min=self.anchor_speeds[0].item(), max=self.anchor_speeds[-1].item())
    
    # Find indices for interpolation
    v_min = self.anchor_speeds[0].item()
    v_max = self.anchor_speeds[-1].item()
    num_anchors = len(self.anchor_speeds)
    
    idx_float = (v_clamped - v_min) / (v_max - v_min) * (num_anchors - 1)
    idx_low = idx_float.floor().long()
    idx_high = (idx_low + 1).clamp(max=num_anchors - 1)
    alpha = (idx_float - idx_low.float()).unsqueeze(1) # [B, 1]
    
    # Interpolate frequencies to find current F_RL
    freq_low = self.anchor_frequencies[idx_low]
    freq_high = self.anchor_frequencies[idx_high]
    current_freq = freq_low + (freq_high - freq_low) * alpha.squeeze(1)
    
    # Advance phase using exact frequency
    phase_delta = self.step_dt * current_freq
    self.gait_phase = torch.fmod(self.gait_phase + phase_delta, 1.0)
    
    # Get frame indices
    frame_idx_float = self.gait_phase * self.num_frames
    frame_idx_low = frame_idx_float.floor().long() % self.num_frames
    frame_idx_high = (frame_idx_low + 1) % self.num_frames
    frame_alpha = (frame_idx_float - frame_idx_float.floor()).unsqueeze(1) # [B, 1]

    # Bilinear interpolation
    traj_vlow_flow = self.anchor_trajectories[idx_low, frame_idx_low]
    traj_vlow_fhigh = self.anchor_trajectories[idx_low, frame_idx_high]
    traj_vhigh_flow = self.anchor_trajectories[idx_high, frame_idx_low]
    traj_vhigh_fhigh = self.anchor_trajectories[idx_high, frame_idx_high]
    
    ref_pos_vlow = traj_vlow_flow + (traj_vlow_fhigh - traj_vlow_flow) * frame_alpha
    ref_pos_vhigh = traj_vhigh_flow + (traj_vhigh_fhigh - traj_vhigh_flow) * frame_alpha
    ref_pos_all = ref_pos_vlow + (ref_pos_vhigh - ref_pos_vlow) * alpha
    
    # Extract matching joint subsets
    ref_pos = ref_pos_all[:, self.traj_indices]
    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids][:, self.sim_indices]
    error_sq = torch.sum(torch.square(current_pos - ref_pos), dim=1)

    dynamic_std = std * (total_speed / 0.5).clamp(min=1.0, max=3.0)
    reward = torch.exp(-error_sq / (dynamic_std ** 2))
    reward = reward * active

    reset_ids = env.termination_manager.terminated.nonzero(as_tuple=False).flatten()
    if len(reset_ids) > 0:
      self.gait_phase[reset_ids] = torch.rand(len(reset_ids), device=env.device)

    return reward


class feet_proximity:
  """Penalize feet from getting too close to each other in 3D space."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    pass

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    min_distance: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    assert len(asset_cfg.site_ids) == 2, "feet_proximity reward requires exactly 2 sites"
    
    pos_left = asset.data.site_pos_w[:, asset_cfg.site_ids[0], :]
    pos_right = asset.data.site_pos_w[:, asset_cfg.site_ids[1], :]
    
    # 3D Euclidean distance
    dist = torch.norm(pos_left - pos_right, dim=-1)
    
    # Quadratic penalty: w * max(0, min_distance - dist)^2
    penalty = torch.square(torch.clamp(min_distance - dist, min=0.0))
    return penalty
