"""Footstep Manager for Angad bipedal robot.

This module implements a fully batched, GPU-native footstep sequencing system
for use with mjlab + mjwarp + rsl_rl. It is the equivalent of Rohan P. Singh's
stepping_task.py, but operates purely on PyTorch tensors instead of physical
MuJoCo geom manipulation (which is not possible with mjwarp's static GPU kernels).

Architecture:
  - Generates a sequence of (x, y, z, theta) footstep targets for each env in
    the batch.
  - Expresses the next two upcoming targets in the robot's local body frame so
    the RL policy can condition on them regardless of world position / heading.
  - Detects when a foot lands within the target radius and advances the pointer
    to the next target.
  - Resets cleanly on episode termination.

Usage in env_cfg.py:
  The FootstepManager is instantiated once (as a stateful observation / reward
  helper) and queried every step from observation and reward term callbacks.
"""

from __future__ import annotations

import math
import os
import random
from typing import TYPE_CHECKING
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT HYPERPARAMETERS FOR FOOTSTEP MANAGER & MULTI-SEGMENT SPLINE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

# ── Gait & Target Footstep Constants ─────────────────────────────────────────
DEFAULT_NUM_STEPS: int = 20              # Base steps count buffer
DEFAULT_STEP_LENGTH: float = 0.28        # Stride half-length in meters
DEFAULT_STEP_WIDTH: float = 0.12         # Lateral foot stance width in meters
DEFAULT_STEP_HEIGHT: float = 0.0         # Vertical step increment in meters (0.0 = flat floor)
DEFAULT_TARGET_RADIUS: float = 0.08      # Touchdown target hit radius in meters
DEFAULT_DELAY_STEPS: int = 3             # Touchdown debounce count (3 x 0.02s = 0.06s)
DEFAULT_GAIT_FREQUENCY: float = 1.0 / 2.20  # Gait clock frequency (≈ 0.4545 Hz, 2.2s period)

# ── Multi-Segment Spline Generation Hyperparameters ───────────────────────────
DEFAULT_NUM_SPLINE_POINTS: int = 4       # Number of control points P1..P4 after pelvis anchor P0
DEFAULT_MIN_SEGMENT_LENGTH: float = 3.0   # Minimum distance between spline waypoints in meters
DEFAULT_MAX_SEGMENT_LENGTH: float = 5.0   # Maximum distance between spline waypoints in meters
DEFAULT_MAX_ANGLE_DEV_DEG: float = 40.0   # Max angular deviation relative to previous segment [+-40 deg]
DEFAULT_STEP_DISTANCE: float = 0.56      # Distance between two consecutive steps on SAME side in meters

# ── Progressive Step-Length Curriculum Hyperparameters ───────────────────────
DEFAULT_ENABLE_STEP_CURRICULUM: bool = True
DEFAULT_MIN_STEP_DISTANCE: float = 0.25   # Initial shuffle step distance in meters (iter <= 500)
DEFAULT_MAX_STEP_DISTANCE: float = 0.60   # Full stride step distance in meters (iter >= 5000)
DEFAULT_CURRICULUM_START_ITER: int = 500  # Iteration where step length curriculum begins scaling
DEFAULT_CURRICULUM_END_ITER: int = 5000   # Iteration where step length curriculum reaches max stride


class FootstepManager:
    """Batched footstep target sequencer — runs entirely on the GPU.

    Each environment in the parallel batch gets its own independent footstep
    sequence.  The sequences are stored as fixed-length tensors to avoid
    Python-side loops during stepping.
    """

    LEFT  = 0  # index into the contact sensor's [B, 2] tensors
    RIGHT = 1

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        num_steps: int = DEFAULT_NUM_STEPS,
        step_length: float = DEFAULT_STEP_LENGTH,
        step_width: float = DEFAULT_STEP_WIDTH,
        step_height: float = DEFAULT_STEP_HEIGHT,
        target_radius: float = DEFAULT_TARGET_RADIUS,
        delay_steps: int = DEFAULT_DELAY_STEPS,
        plan_path: str | None = None,
        forced_mode: str | int | None = None,
        num_spline_points: int = DEFAULT_NUM_SPLINE_POINTS,
        step_distance: float = DEFAULT_STEP_DISTANCE,
        min_segment_length: float = DEFAULT_MIN_SEGMENT_LENGTH,
        max_segment_length: float = DEFAULT_MAX_SEGMENT_LENGTH,
        max_angle_dev_deg: float = DEFAULT_MAX_ANGLE_DEV_DEG,
        lateral_offset: float | None = None,
        enable_step_curriculum: bool = DEFAULT_ENABLE_STEP_CURRICULUM,
        min_step_distance: float = DEFAULT_MIN_STEP_DISTANCE,
        max_step_distance: float = DEFAULT_MAX_STEP_DISTANCE,
        curriculum_start_iter: int = DEFAULT_CURRICULUM_START_ITER,
        curriculum_end_iter: int = DEFAULT_CURRICULUM_END_ITER,
    ):
        self.env = env
        self.B = env.num_envs
        self.device = env.device

        self.num_steps    = num_steps
        self.step_length  = step_length
        self.step_width   = step_width
        self.step_height  = step_height
        self.target_radius = target_radius
        self.delay_steps   = delay_steps
        self.plan_path     = plan_path

        # ── Spline Trajectory Hyperparameters ────────────────────────────────
        self.num_spline_points        = num_spline_points
        self.step_distance            = step_distance       # distance between 2 steps on SAME side
        self.min_segment_length       = min_segment_length  # min distance between spline points [3m]
        self.max_segment_length       = max_segment_length  # max distance between spline points [5m]
        self.max_angle_dev_deg        = max_angle_dev_deg   # max angular deviation [+-40 deg]
        self.custom_lateral_offset     = lateral_offset
        self.enable_step_curriculum   = enable_step_curriculum
        self.min_step_distance        = min_step_distance
        self.max_step_distance        = max_step_distance
        self.curriculum_start_iter    = curriculum_start_iter
        self.curriculum_end_iter      = curriculum_end_iter

        # Dynamically compute max steps required to cover the full spline path even at smallest step distance
        d_stride_min = max(0.05, min_step_distance / 2.0)
        calc_max_steps = math.ceil((num_spline_points * max_segment_length) / d_stride_min) + 10
        self.num_steps = max(num_steps, calc_max_steps)

        # Mode mapping dictionary
        self.mode_map = {
            "curved_left": 0, "curved": 0, 0: 0,
            "standing": 1, 1: 1,
            "backward": 2, "back": 2, 2: 2,
            "lateral": 3, 3: 3,
            "forward": 4, "front": 4, 4: 4,
            "curved_right": 5, 5: 5,
        }
        self.forced_mode_idx = None
        if forced_mode is not None:
            m_key = str(forced_mode).lower() if isinstance(forced_mode, str) else forced_mode
            if m_key in self.mode_map:
                self.forced_mode_idx = self.mode_map[m_key]
                print(f"[INFO] FootstepManager forced walk mode: {forced_mode} (index {self.forced_mode_idx})")

        # ── Load custom plan if provided ─────────────────────────────────────
        self.loaded_plan = None
        if plan_path is not None and os.path.exists(plan_path):
            try:
                # Load x, y, z, theta from file (skip header)
                data = np.loadtxt(plan_path, delimiter=",", skiprows=1)
                # Keep only up to num_steps
                if len(data) > self.num_steps:
                    data = data[:self.num_steps]
                elif len(data) < self.num_steps:
                    # Pad with last step if plan is shorter than num_steps
                    padding = np.tile(data[-1], (self.num_steps - len(data), 1))
                    data = np.concatenate([data, padding], axis=0)
                
                # Format: [num_steps, 4] -> (x, y, z, theta)
                self.loaded_plan = torch.tensor(data[:, :4], device=self.device, dtype=torch.float32)
                print(f"[INFO] FootstepManager loaded plan from {plan_path} with {self.num_steps} steps.")
            except Exception as e:
                print(f"[ERROR] Failed to load footstep plan from {plan_path}: {e}")

        # ── Load curved footstep plans (relative local fallback) ──────────────
        self.plans = []
        local_plans_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "footstep_plans.txt")
        rohan_plans_path = "/home/nandhith/Python/Humanoid_Xterra_IITK/Rohanpsingh_Trials/utils/footstep_plans.txt"
        target_path = local_plans_path if os.path.exists(local_plans_path) else rohan_plans_path
        if os.path.exists(target_path):
            try:
                with open(target_path) as fn:
                    lines = [ln.strip() for ln in fn.readlines()]
                sequence = []
                for line in lines:
                    if line == "---":
                        if len(sequence):
                            self.plans.append(sequence)
                        sequence = []
                    else:
                        sequence.append(np.array([float(val) for val in line.split(",")]))
                print(f"[INFO] FootstepManager loaded {len(self.plans)} curved plans from {target_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load curved plans: {e}")

        self.step_counter = 0

        # ── Persistent state tensors (all shape [B, ...]) ─────────────────
        # World-frame footstep sequence: [B, num_steps, 4]  (x, y, z, theta)
        self.sequence = torch.zeros(
            (self.B, self.num_steps, 4), device=self.device, dtype=torch.float32
        )

        # Current active target indices for "next" (t1) and "lookahead" (t2)
        self.t1 = torch.zeros(self.B, device=self.device, dtype=torch.long)
        self.t2 = torch.ones(self.B, device=self.device, dtype=torch.long)

        # Debounce counter: how many consecutive steps has the foot been in radius
        self.in_radius_count = torch.zeros(self.B, device=self.device, dtype=torch.long)
        self.steps_since_hit = torch.zeros(self.B, device=self.device, dtype=torch.long)

        # Centralized gait clock phase tensor [B]: synchronized for observations and rewards
        self.gait_frequency: float = 1.0 / 2.20  # ≈ 0.4545 Hz, matches Rohan exactly
        self.gait_phase = torch.zeros(self.B, device=self.device, dtype=torch.float32)

        # Gait mode: 0=WALKING (alternating), 1=STANDING (both feet grounded)
        self.gait_mode = torch.zeros(self.B, device=self.device, dtype=torch.long)

        # Whether the first target step is for the left foot (static throughout episode)
        self.first_step_is_left = torch.zeros(self.B, device=self.device, dtype=torch.bool)

        # Active steps count per environment
        self.num_active_steps = torch.zeros(self.B, device=self.device, dtype=torch.long)

        # Spline control points P0..PN [B, num_spline_points + 1, 3]
        self.control_points = torch.zeros((self.B, self.num_spline_points + 1, 3), device=self.device, dtype=torch.float32)

        # Spline curve dense samples [B, num_steps, 3] for 3D visualization
        self.spline_points = torch.zeros((self.B, self.num_steps, 3), device=self.device, dtype=torch.float32)

        # Whether the current episode's sequence has been initialised
        self.initialised = torch.zeros(self.B, device=self.device, dtype=torch.bool)

        # Generate the initial sequences for every environment
        self._generate_all_sequences()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update_curriculum(self, current_iter: int) -> None:
        """Update 2-axis training curriculum parameters based on current RL iteration count.

        Axis 1 (Path Complexity):
            Iter 0 -> 3000:
                - Segment distance range contracts: [4.0-5.0m] -> [1.5-3.0m]
                - Max angular heading turn expands: ±5 deg -> ±30 deg

        Axis 2 (Gait Speed / Swing Time Decay):
            Iter 0 -> 5000:
                - Gait cycle time contracts: 2.4s -> 1.2s per step
                - Gait clock frequency increases: 0.416 Hz -> 0.833 Hz
        """
        # Axis 1: Path Complexity (0 -> 3000 iterations)
        path_alpha = min(1.0, max(0.0, float(current_iter) / 3000.0))
        self.max_angle_dev_deg = 0.0 + path_alpha * 30.0  # 0 deg (100% straight path at start) -> 30 deg
        self.min_segment_length = 5.0 - path_alpha * 3.5  # 5.0m -> 1.5m
        self.max_segment_length = 5.0 - path_alpha * 2.0  # 5.0m -> 3.0m

        # Axis 2: Gait Speed / Swing Duration (0 -> 5000 iterations)
        speed_alpha = min(1.0, max(0.0, float(current_iter) / 5000.0))
        gait_period = 2.4 - speed_alpha * 1.2             # 2.4s -> 1.2s per step
        self.gait_frequency = 1.0 / gait_period

    def get_active_target_is_left(self) -> torch.Tensor:
        """Return boolean tensor [B] indicating if current t1 target is for the left foot."""
        t1_is_even = (self.t1 % 2 == 0)
        return torch.where(self.first_step_is_left, t1_is_even, ~t1_is_even)

    def get_swing_phase(self) -> torch.Tensor:
        """Return phase normalized to [0, 1) within the CURRENT active swing
        (right-swing window [0,0.5) or left-swing window [0.5,1.0)), matching
        what compute_reference_joints / compute_dynamic_com expect as `s`."""
        return torch.where(
            self.gait_phase < 0.5,
            self.gait_phase * 2.0,
            (self.gait_phase - 0.5) * 2.0,
        )

    def step(
        self,
        left_foot_pos_w: torch.Tensor,   # [B, 3]  world-frame left  foot pos
        right_foot_pos_w: torch.Tensor,  # [B, 3]  world-frame right foot pos
    ) -> None:
        """Advance the footstep pointer if the correct foot has landed on the target.

        Call this once per RL step, before querying get_goal_steps_local().

        Args:
            left_foot_pos_w:  world-frame position of the left  foot  [B, 3]
            right_foot_pos_w: world-frame position of the right foot  [B, 3]
        """
        # Gather current t1 target position: [B, 3]
        t1_pos = self._gather_target_pos(self.t1)  # [B, 3]

        # Distance from each foot to current t1 target (kept for hit_rate logging)
        left_dist  = torch.norm(left_foot_pos_w  - t1_pos, dim=1)  # [B]
        right_dist = torch.norm(right_foot_pos_w - t1_pos, dim=1)  # [B]
        t1_is_left = self.get_active_target_is_left()
        target_dist = torch.where(t1_is_left, left_dist, right_dist)

        # ── Time-based Target Progression ──
        # Instead of waiting for the physical foot to land (which desynchronizes from the continuous gait_phase clock),
        # we advance the target pointers strictly when the gait_phase crosses 0.5 (half cycle) or 0.0 (full cycle wrap).
        # This guarantees 100% synchronization between foot_clock_reward commands and the presented targets.
        
        old_phase = self.gait_phase.clone()
        
        # Advance centralized gait phase
        self.gait_phase = torch.fmod(
            self.gait_phase + self.env.step_dt * self.gait_frequency, 1.0
        )
        
        # Check if phase crossed 0.5 boundary
        crossed_half = (old_phase < 0.5) & (self.gait_phase >= 0.5)
        # Check if phase crossed 1.0 boundary (wrapped to 0.0)
        crossed_wrap = (old_phase > 0.5) & (self.gait_phase < 0.5)
        
        should_advance = crossed_half | crossed_wrap

        # Record hit statistics right before target advances
        if hasattr(self.env, "extras") and "log" in self.env.extras:
            if should_advance.any():
                hit_dist = target_dist[should_advance].mean().item()
                hit_rate = (target_dist[should_advance] < self.target_radius).float().mean().item()
                self.env.extras["log"]["Footstep/mean_foot_to_target_dist"] = hit_dist
                self.env.extras["log"]["Footstep/hit_rate"] = hit_rate

        # Advance the target pointer strictly on the clock
        self._advance_targets(should_advance, sync_clock=False)

        self.step_counter += 1

    def get_goal_steps_local(
        self,
        root_pos_w: torch.Tensor,   # [B, 3]
        root_quat_w: torch.Tensor,  # [B, 4]  (w, x, y, z)
    ) -> torch.Tensor:
        """Return next-two footstep targets expressed in the robot's body frame.

        Returns:
            Tensor of shape [B, 8]:
              [t1_x, t1_y, t1_z, t1_theta, t2_x, t2_y, t2_z, t2_theta]
            all values relative to the robot's current base pose.
        """
        t1_world = self._gather_target(self.t1)  # [B, 4]
        t2_world = self._gather_target(self.t2)  # [B, 4]

        t1_local = self._world_to_local(t1_world, root_pos_w, root_quat_w)
        t2_local = self._world_to_local(t2_world, root_pos_w, root_quat_w)

        return torch.cat([t1_local, t2_local], dim=1)  # [B, 8]

    def get_target_pos_world(self) -> torch.Tensor:
        """Return the current t1 target position in world frame. [B, 3]"""
        return self._gather_target_pos(self.t1)

    def get_target_midpoint_world(self) -> torch.Tensor:
        """Return the midpoint between t1 and t2 targets in world frame. [B, 3]"""
        t1_pos = self._gather_target_pos(self.t1)
        t2_pos = self._gather_target_pos(self.t2)
        return (t1_pos + t2_pos) * 0.5  # [B, 3]

    @property
    def iteration_count(self) -> int:
        """Return the approximate PPO training iteration count."""
        return self.step_counter // 24

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset footstep sequences for the given environment indices.

        Called by the environment on episode termination.

        Args:
            env_ids: 1-D tensor of environment indices to reset.
        """
        if len(env_ids) == 0:
            return

        self.t1[env_ids] = 0
        self.t2[env_ids] = 1
        self.in_radius_count[env_ids] = 0
        self.steps_since_hit[env_ids] = 0

        # Rohan: only randomize to phase 0.0 (right starts swinging) or 0.5 (left starts)
        # This guarantees clean starts at phase boundaries, not mid-swing
        rand_bits = torch.randint(0, 2, (len(env_ids),), device=self.device).float()
        self.gait_phase[env_ids] = rand_bits * 0.5

        # Rohan's WalkModes: CURVED (15%), STANDING (5%), BACKWARD (20%), LATERAL (30%), FORWARD (30%)
        if self.forced_mode_idx is not None:
            self.gait_mode[env_ids] = self.forced_mode_idx
        else:
            mode_probs = torch.tensor([0.15, 0.05, 0.20, 0.30, 0.30], device=self.device)
            self.gait_mode[env_ids] = torch.multinomial(mode_probs, len(env_ids), replacement=True)

        self.initialised[env_ids] = False

        # Regenerate fresh sequences for these envs
        self._generate_sequences_for(env_ids)
        self.initialised[env_ids] = True

    # ──────────────────────────────────────────────────────────────────────────
    # Sequence Generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_all_sequences(self) -> None:
        """Generate footstep sequences for every environment at startup."""
        all_ids = torch.arange(self.B, device=self.device)
        self._generate_sequences_for(all_ids)
        self.initialised[:] = True

    def _generate_sequences_for(self, env_ids: torch.Tensor) -> None:
        """Generate or load the footstep sequence for the given environments.

        Alternates left/right feet. If self.loaded_plan is present, rotates and
        translates the plan relative to the robot's spawn coordinate. Otherwise,
        generates a straight line forward.
        """
        n = len(env_ids)
        if n == 0:
            return

        import os
        import numpy as np
        import random

        robot = self.env.scene["robot"]
        spawn_pos  = robot.data.root_link_pos_w[env_ids]   # [n, 3]
        spawn_quat = robot.data.root_link_quat_w[env_ids]  # [n, 4]
        spawn_yaw  = _quat_to_yaw(spawn_quat)              # [n]
        
        # Determine stance lateral offset (default 0.12m = 12cm per side -> 24cm full stance width)
        if self.custom_lateral_offset is not None:
            stance_w = self.custom_lateral_offset
        else:
            stance_w = self.step_width  # 0.12m offset per side for natural biped stance width

        # Determine dynamic step_distance from training curriculum if enabled
        if self.enable_step_curriculum:
            iter_cnt = getattr(self, "iteration_count", 0)
            iter_range = max(1.0, float(self.curriculum_end_iter - self.curriculum_start_iter))
            h = float(np.clip((iter_cnt - self.curriculum_start_iter) / iter_range, 0.0, 1.0))
            curr_step_dist = self.min_step_distance + h * (self.max_step_distance - self.min_step_distance)
        else:
            curr_step_dist = self.step_distance

        # Step distance between same-side steps -> stride along curve per foot step is curr_step_dist / 2
        d_stride = max(0.05, curr_step_dist / 2.0)

        # ── 1. Fast Vectorized Spline Control Points P0..PN [n, N+1, 2] ──────────
        # Gait mode: 0=curved, 1=standing, 2=backward, 3=lateral, 4=forward, 5=curved_right
        gait_mode_env = self.gait_mode[env_ids]  # [n]

        p0 = spawn_pos[:, :2]  # [n, 2]
        pts_list = [p0]

        # Base heading direction per gait mode:
        #   backward (2): face away from spawn (pi offset)
        #   lateral (3): face 90-deg left from spawn heading
        #   forward (4): face spawn heading exactly
        #   curved left (0): spawn heading + slight left bias
        #   curved right (5): spawn heading + slight right bias
        base_heading = spawn_yaw.clone()  # [n]
        is_backward = (gait_mode_env == 2)
        is_lateral_l = (gait_mode_env == 3)
        is_lateral_r = (gait_mode_env == 5)

        # Adjust base heading per mode
        base_heading = torch.where(is_backward,  base_heading + math.pi, base_heading)
        base_heading = torch.where(is_lateral_l, base_heading + math.pi / 2, base_heading)
        base_heading = torch.where(is_lateral_r, base_heading - math.pi / 2, base_heading)
        curr_heading = base_heading.clone()

        # Restrict angle deviation for lateral/backward (tight corridor)
        is_special = is_backward | is_lateral_l | is_lateral_r
        max_dev_rad_normal  = math.radians(self.max_angle_dev_deg)
        max_dev_rad_special = math.radians(10.0)  # tight corridor for non-forward modes

        for k in range(1, self.num_spline_points + 1):
            seg_len = torch.empty(n, device=self.device).uniform_(self.min_segment_length, self.max_segment_length)
            # Use tight deviation for backward/lateral modes
            max_dev = torch.where(is_special,
                torch.full((n,), max_dev_rad_special, device=self.device),
                torch.full((n,), max_dev_rad_normal,  device=self.device))
            angle_dev = (torch.rand(n, device=self.device) * 2.0 - 1.0) * max_dev
            curr_heading = curr_heading + angle_dev
            pk = pts_list[-1] + seg_len.unsqueeze(-1) * torch.stack([torch.cos(curr_heading), torch.sin(curr_heading)], dim=-1)
            pts_list.append(pk)

        pts_tensor = torch.stack(pts_list, dim=1)  # [n, N+1, 2]

        # Store 3D control points P0..PN for visualization [n, N+1, 3]
        ctrl_3d = torch.zeros((n, self.num_spline_points + 1, 3), device=self.device, dtype=torch.float32)
        ctrl_3d[:, :, :2] = pts_tensor
        ctrl_3d[:, :, 2] = 0.01
        self.control_points[env_ids] = ctrl_3d

        # ── 2. Cumulative Chordal Segment Distances & Active Step Counts ──────
        seg_dists = torch.norm(pts_tensor[:, 1:] - pts_tensor[:, :-1], dim=-1)  # [n, N]
        s_knots = torch.cat([torch.zeros((n, 1), device=self.device), torch.cumsum(seg_dists, dim=-1)], dim=-1)  # [n, N+1]
        L_total = s_knots[:, -1]  # [n]

        n_path_steps = torch.clamp(torch.ceil(L_total / d_stride).long(), min=1)  # [n]
        self.num_active_steps[env_ids] = n_path_steps

        env_phase = self.gait_phase[env_ids]
        first_is_left = (env_phase == 0.5)
        self.first_step_is_left[env_ids] = first_is_left

        # ── 3. Vectorized Step Evaluation along Multi-Segment Spline ──────────
        step_indices = torch.arange(self.num_steps, device=self.device).unsqueeze(0).expand(n, -1)  # [n, num_steps]
        s_vals = (step_indices.float() + 1.0) * d_stride  # [n, num_steps]

        # Map step distances s_vals to segment index k
        s_expanded = s_vals.unsqueeze(-1)  # [n, num_steps, 1]
        k_mask = (s_expanded >= s_knots[:, :-1].unsqueeze(1)) & (s_expanded <= s_knots[:, 1:].unsqueeze(1))  # [n, num_steps, N]
        seg_idx = k_mask.long().argmax(dim=-1)  # [n, num_steps]

        # Gather start/end control points for each step's segment
        p_start = torch.gather(pts_tensor[:, :-1].unsqueeze(1).expand(-1, self.num_steps, -1, -1), 2, seg_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(2)  # [n, num_steps, 2]
        p_end   = torch.gather(pts_tensor[:, 1:].unsqueeze(1).expand(-1, self.num_steps, -1, -1), 2, seg_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(2)    # [n, num_steps, 2]
        s_start = torch.gather(s_knots[:, :-1].unsqueeze(1).expand(-1, self.num_steps, -1), 2, seg_idx.unsqueeze(-1)).squeeze(2)  # [n, num_steps]
        s_end   = torch.gather(s_knots[:, 1:].unsqueeze(1).expand(-1, self.num_steps, -1), 2, seg_idx.unsqueeze(-1)).squeeze(2)    # [n, num_steps]

        seg_len_step = (s_end - s_start).clamp(min=1e-5)  # [n, num_steps]
        u_step = ((s_vals - s_start) / seg_len_step).clamp(0.0, 1.0)  # [n, num_steps]

        # Interpolated spline center positions [n, num_steps, 2]
        cx = p_start[:, :, 0] + u_step * (p_end[:, :, 0] - p_start[:, :, 0])
        cy = p_start[:, :, 1] + u_step * (p_end[:, :, 1] - p_start[:, :, 1])

        # Segment tangent directions
        dx = (p_end[:, :, 0] - p_start[:, :, 0]) / seg_len_step
        dy = (p_end[:, :, 1] - p_start[:, :, 1]) / seg_len_step
        heading = torch.atan2(dy, dx)  # [n, num_steps]

        # Alternating left/right foot offset
        is_even_step = (step_indices % 2 == 0)
        is_left_foot = torch.where(first_is_left.unsqueeze(-1), is_even_step, ~is_even_step)  # [n, num_steps]
        y_side = torch.where(is_left_foot, 1.0, -1.0)  # [n, num_steps]

        # Normal vector (-sin theta, cos theta)
        nx = -torch.sin(heading)
        ny =  torch.cos(heading)

        fx = cx + y_side * stance_w * nx
        fy = cy + y_side * stance_w * ny

        # Construct final sequence tensor [n, num_steps, 4]
        seq_tensor = torch.stack([fx, fy, torch.zeros_like(fx), heading], dim=-1)
        self.sequence[env_ids] = seq_tensor

        # ── 4. Continuous Spline Visualization Points [n, num_steps, 3] ──────
        sp_pts = torch.zeros((n, self.num_steps, 3), device=self.device, dtype=torch.float32)
        sp_pts[:, :, 0] = cx
        sp_pts[:, :, 1] = cy
        sp_pts[:, :, 2] = 0.01
        self.spline_points[env_ids] = sp_pts

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _gather_target(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather (x, y, z, theta) for each env from `indices`. Returns [B, 4]."""
        idx = indices.clamp(max=self.num_steps - 1).unsqueeze(1).unsqueeze(2)  # [B,1,1]
        idx = idx.expand(-1, 1, 4)                                              # [B,1,4]
        return self.sequence.gather(1, idx).squeeze(1)                         # [B, 4]

    def _gather_target_pos(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather (x, y, z) position for each env. Returns [B, 3]."""
        return self._gather_target(indices)[:, :3]  # [B, 3]

    def _advance_targets(self, mask: torch.Tensor, sync_clock: bool = True) -> None:
        """Advance t1 → t2 and t2 → t2+1 for envs where mask is True."""
        max_idx = self.num_steps - 1
        self.t1 = torch.where(mask, self.t2,                          self.t1)
        self.t2 = torch.where(mask, (self.t2 + 1).clamp(max=max_idx), self.t2)
        
        if sync_clock:
            # Synchronize gait clock: each footstep hit = half-cycle elapsed
            self.gait_phase = torch.where(
                mask,
                torch.fmod(self.gait_phase + 0.5, 1.0),
                self.gait_phase,
            )

    def _world_to_local(
        self,
        target_world: torch.Tensor,   # [B, 4]  (x, y, z, theta)
        root_pos_w: torch.Tensor,     # [B, 3]
        root_quat_w: torch.Tensor,    # [B, 4]  (w, x, y, z)
    ) -> torch.Tensor:
        """Express target in robot body frame. Returns [B, 4]."""
        root_yaw = _quat_to_yaw(root_quat_w)  # [B]

        dx = target_world[:, 0] - root_pos_w[:, 0]
        dy = target_world[:, 1] - root_pos_w[:, 1]
        dz = target_world[:, 2] - root_pos_w[:, 2]

        cos_yaw = torch.cos(-root_yaw)
        sin_yaw = torch.sin(-root_yaw)

        x_local = cos_yaw * dx - sin_yaw * dy
        y_local = sin_yaw * dx + cos_yaw * dy
        z_local = dz

        # Relative heading: wrap to [-pi, pi]
        d_theta = target_world[:, 3] - root_yaw
        d_theta = (d_theta + math.pi) % (2 * math.pi) - math.pi

        return torch.stack([x_local, y_local, z_local, d_theta], dim=1)  # [B, 4]

    def debug_vis(self, visualizer: "DebugVisualizer") -> None:
        """Render target footsteps, control points, and spline curve in GUI.

        Called automatically by the env's visualizer update loop.
        """
        import numpy as np
        
        env_idx = visualizer.env_idx

        # ── 1. Render Control Points P0..PN ──────────────────────────────────
        ctrl_pts = self.control_points[env_idx].cpu().numpy()
        colors_palette = [
            (0.2, 0.9, 0.3, 0.95),  # P0: Bright Green
            (1.0, 0.85, 0.0, 0.95), # P1: Bright Gold
            (0.85, 0.1, 0.95, 0.95),# P2: Bright Magenta
            (1.0, 0.4, 0.0, 0.95),  # P3: Bright Orange
            (0.1, 0.9, 0.9, 0.95),  # P4: Cyan
        ]
        for idx_p, pt in enumerate(ctrl_pts):
            c_color = colors_palette[idx_p % len(colors_palette)]
            visualizer.add_sphere(
                center=pt,
                radius=0.06,  # 3x smaller (was 0.18)
                color=c_color,
                label=f"P{idx_p}",
            )

        # ── 2. Render Continuous Spline Curve Path (Cyan Line Spheres) ───────
        spline_pts = self.spline_points[env_idx].cpu().numpy()
        for j in range(len(spline_pts)):
            visualizer.add_sphere(
                center=spline_pts[j],
                radius=0.012,  # 3x smaller (was 0.035)
                color=(0.1, 0.9, 1.0, 0.8),
                label=f"spline_{j}",
            )

        # ── 3. Render Footstep Target Spheres & Tangent-Parallel Arrows ──────
        t1_idx = self.t1[env_idx].item()
        t2_idx = self.t2[env_idx].item()
        n_steps_to_draw = self.num_active_steps[env_idx].item()

        for i in range(n_steps_to_draw):
            x, y, z, theta = self.sequence[env_idx, i].cpu().tolist()

            first_is_left = self.first_step_is_left[env_idx].item()
            is_left_foot = (i % 2 == 0) if first_is_left else (i % 2 != 0)

            if is_left_foot:
                base_color = (0.1, 0.4, 1.0)   # Bright Blue (Left Foot)
            else:
                base_color = (1.0, 0.15, 0.15)  # Bright Red (Right Foot)

            if i == t1_idx:
                radius = 0.026  # 3x smaller (was 0.08)
                color = (base_color[0], base_color[1], base_color[2], 0.95)
            elif i == t2_idx:
                radius = 0.022  # 3x smaller (was 0.065)
                color = (base_color[0], base_color[1], base_color[2], 0.80)
            else:
                radius = 0.016  # 3x smaller (was 0.050)
                color = (base_color[0], base_color[1], base_color[2], 0.50)

            visualizer.add_sphere(
                center=np.array([x, y, z + 0.01]),
                radius=radius,
                color=color,
                label=f"step_{i}",
            )

            # Draw orientation arrow pointing parallel to spline tangent
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            visualizer.add_arrow(
                start=np.array([x, y, z + 0.01]),
                end=np.array([x + 0.15 * cos_t, y + 0.15 * sin_t, z + 0.01]),
                color=color,
                width=0.01,
                label=f"step_arrow_{i}",
            )



# ── Utility ───────────────────────────────────────────────────────────────────

def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw (rotation about Z) from a (w, x, y, z) quaternion. [B] → [B]."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)
