"""PyTorch Vectorized Kinematic Trajectory Generator for Angad Humanoid.

Computes the reference joint trajectory q_ref(phi) on-the-fly for any batch of
footstep targets and gait phases phi in [0, 1).

Matches the exact arc + flat foot + pelvis shift kinematics of kinematic_walk_demo.py.
Runs on PyTorch CUDA tensors for batch size N (e.g. 4096 parallel envs).
"""

import math
import torch

# Default joint keyframe angles for Stable_Stance pose
# 12 joints: [right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee_pitch, right_ankle_pitch, right_ankle_roll,
#            left_hip_pitch,  left_hip_roll,  left_hip_yaw,  left_knee_pitch,  left_ankle_pitch,  left_ankle_roll]
STABLE_STANCE_QPOS = torch.tensor([
    -0.45, 0.0, 0.0, 0.90, -0.45, 0.0,
    -0.45, 0.0, 0.0, 0.90, -0.45, 0.0,
], dtype=torch.float32)

# Angad link lengths (m)
FEMUR_LEN = 0.1566
TIBIA_LEN = 0.2097
PELVIS_HEIGHT = 0.831
FOOT_SITE_Z = 0.0644

class KinematicTrajectoryGenerator:
    """Vectorized PyTorch Kinematic Reference Trajectory Generator."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.q_default = STABLE_STANCE_QPOS.to(device).unsqueeze(0).repeat(num_envs, 1)

    def compute_reference_joints(
        self,
        gait_phase: torch.Tensor,      # [num_envs] phase in [0, 1) per step
        active_foot: torch.Tensor,     # [num_envs] 0 = Left swing, 1 = Right swing
        p_stance: torch.Tensor,        # [num_envs, 3] world pos of stance foot
        p_swing_start: torch.Tensor,   # [num_envs, 3] world pos of swing foot start
        t1_target: torch.Tensor,       # [num_envs, 4] target (x, y, z, theta) for active swing foot
        arc_height: float = 0.12,
        p_pelvis_override: torch.Tensor | None = None, # Optional: LIPM CoM override
    ) -> torch.Tensor:
        """Computes reference joint angles q_ref for all envs.

        Returns:
            q_ref: [num_envs, 12] target joint angles matching kinematic arc walk.
        """
        # Linear phase s in [0, 1] within the step
        s = gait_phase.clamp(0.0, 1.0).unsqueeze(-1)  # [num_envs, 1]

        # 1. Swing foot position along semicircular arc
        p_swing_xy = (1.0 - s) * p_swing_start[:, :2] + s * t1_target[:, :2]
        p_swing_z  = (1.0 - s) * p_swing_start[:, 2:3] + s * FOOT_SITE_Z + arc_height * torch.sin(math.pi * s)
        p_swing = torch.cat([p_swing_xy, p_swing_z], dim=-1)  # [num_envs, 3]

        # 2. Pelvis position shift (use LIPM CoM override if provided to avoid Bug 1 conflict)
        if p_pelvis_override is not None:
            p_pelvis = p_pelvis_override
        else:
            p_pelvis_start = 0.5 * (p_stance[:, :2] + p_swing_start[:, :2])
            p_pelvis_target = 0.5 * (p_stance[:, :2] + t1_target[:, :2])
            p_pelvis_xy = (1.0 - s) * p_pelvis_start + s * p_pelvis_target
            p_pelvis = torch.cat([p_pelvis_xy, torch.full_like(p_swing_z, PELVIS_HEIGHT)], dim=-1)  # [num_envs, 3]

        # 3. Simple analytical 2-link leg pitch calculation (knee/hip pitch)
        # Vector from pelvis to foot site
        r_left  = torch.where(active_foot.unsqueeze(-1) == 0, p_swing - p_pelvis, p_stance - p_pelvis)
        r_right = torch.where(active_foot.unsqueeze(-1) == 1, p_swing - p_pelvis, p_stance - p_pelvis)

        # Leg extension lengths (clamped to valid IK range)
        d_left  = torch.norm(r_left, dim=-1, keepdim=True).clamp(0.20, FEMUR_LEN + TIBIA_LEN - 0.01)
        d_right = torch.norm(r_right, dim=-1, keepdim=True).clamp(0.20, FEMUR_LEN + TIBIA_LEN - 0.01)

        # Law of cosines for knee pitch
        cos_knee_l = (FEMUR_LEN**2 + TIBIA_LEN**2 - d_left**2) / (2.0 * FEMUR_LEN * TIBIA_LEN)
        cos_knee_r = (FEMUR_LEN**2 + TIBIA_LEN**2 - d_right**2) / (2.0 * FEMUR_LEN * TIBIA_LEN)
        knee_pitch_l = math.pi - torch.acos(cos_knee_l.clamp(-0.99, 0.99))
        knee_pitch_r = math.pi - torch.acos(cos_knee_r.clamp(-0.99, 0.99))

        # Hip and ankle pitch calculations incorporating forward/backward lean (Bug 2 fix)
        pitch_ratio_l = (-r_left[:, 2:3] / d_left).clamp(-0.99, 0.99)
        pitch_ratio_r = (-r_right[:, 2:3] / d_right).clamp(-0.99, 0.99)
        pitch_offset_l = torch.asin(pitch_ratio_l)
        pitch_offset_r = torch.asin(pitch_ratio_r)

        hip_pitch_l   = -0.5 * knee_pitch_l + pitch_offset_l * 0.5
        hip_pitch_r   = -0.5 * knee_pitch_r + pitch_offset_r * 0.5
        ankle_pitch_l = -0.5 * knee_pitch_l - pitch_offset_l * 0.5
        ankle_pitch_r = -0.5 * knee_pitch_r - pitch_offset_r * 0.5

        # Roll references zeroed out (holding neutral stance roll)
        hip_roll_l   = torch.zeros_like(hip_pitch_l)
        hip_roll_r   = torch.zeros_like(hip_pitch_r)
        ankle_roll_l = torch.zeros_like(ankle_pitch_l)
        ankle_roll_r = torch.zeros_like(ankle_pitch_r)

        # Heading (yaw) adjustments — zeroed out (Bug 3 fix: eliminates world-frame yaw torque spikes)
        yaw_l = torch.zeros_like(s)
        yaw_r = torch.zeros_like(s)

        # Assemble q_ref [num_envs, 12]
        # Order: [R_hp, R_hr, R_hy, R_kp, R_ap, R_ar, L_hp, L_hr, L_hy, L_kp, L_ap, L_ar]
        q_ref = torch.cat([
            hip_pitch_r, hip_roll_r, yaw_r, knee_pitch_r, ankle_pitch_r, ankle_roll_r,
            hip_pitch_l, hip_roll_l, yaw_l, knee_pitch_l, ankle_pitch_l, ankle_roll_l,
        ], dim=-1)

        return q_ref

