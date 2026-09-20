"""Vectorized PyTorch LIPM & DCM (Divergent Component of Motion) Trajectory Generator.

This module computes dynamically-balanced Center of Mass (CoM) and Zero-Moment Point (ZMP)
trajectories for Angad across 4096 parallel environments on GPU using closed-form
Linear Inverted Pendulum Model (LIPM) and Divergent Component of Motion (DCM / Capture Point)
analytical solutions.

Key Physics & Formulation:
  - 3D Linear Inverted Pendulum:
      x_ddot = (g / z_c) * (x - z_x)
      y_ddot = (g / z_c) * (y - z_y)
      where natural frequency omega = sqrt(g / z_c)

  - DCM / Capture Point:
      xi = x + x_dot / omega
      xi_dot = omega * (xi - z_ref)

  - Backward-Forward Analytical Integration:
      1. Boundary DCM at End of Step (EOS):
         xi_eos = z_target + (z_next - z_target) / (exp(omega * T_step) - 1)
      2. Initial DCM at Start of Step:
         xi_ini = z_stance + (xi_eos - z_stance) * exp(-omega * T_step)
      3. Continuous DCM at time t:
         xi(t) = z_stance + (xi_ini - z_stance) * exp(omega * t)
      4. Continuous Dynamic CoM at time t:
         x_com(t) = z_stance + (x_0 - z_stance) * exp(-omega * t) + (xi_ini - z_stance) * sinh(omega * t)

  - Zero-Moment Point (ZMP) Constraint:
      Mathematically guarantees that the net ground reaction wrench passes through
      the support foot polygon during single-support, eliminating tipping moments.
"""

from __future__ import annotations

import math
from typing import Tuple
import torch

# Standard Gravity and Angad Nominal Walking Height
GRAVITY = 9.81
DEFAULT_COM_HEIGHT = 0.831  # Pelvis height in meters
DEFAULT_STEP_TIME = 1.10    # Step duration in seconds (half-cycle of 2.2s gait period)


class LipmDcmTrajectoryGenerator:
    """Vectorized PyTorch Dynamic LIPM + DCM Trajectory Planner."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        com_height: float = DEFAULT_COM_HEIGHT,
        step_time: float = DEFAULT_STEP_TIME,
        dsp_ratio: float = 0.20,  # 20% Double Support Phase
    ):
        self.num_envs = num_envs
        self.device = device
        self.com_height = com_height
        self.step_time = step_time
        self.dsp_ratio = dsp_ratio

        # Pendulum natural frequency: omega = sqrt(g / z_c)
        self.omega = math.sqrt(GRAVITY / self.com_height)
        self.exp_omega_T = math.exp(self.omega * self.step_time)

    def compute_dynamic_com(
        self,
        gait_phase: torch.Tensor,       # [B] Normalized phase s in [0, 1)
        p_stance: torch.Tensor,         # [B, 3] World pos of stance foot
        p_swing_start: torch.Tensor,    # [B, 3] World pos of swing foot start
        t1_target: torch.Tensor,        # [B, 4] Target 1 (x, y, z, theta) for current swing
        t2_target: torch.Tensor,        # [B, 4] Target 2 (x, y, z, theta) upcoming
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the continuous dynamically-balanced CoM position, CoM velocity, and reference ZMP.

        Args:
            gait_phase: Phase s in [0, 1) within the active step.
            p_stance: Stance foot position [B, 3].
            p_swing_start: Swing foot starting position [B, 3].
            t1_target: Active landing target [B, 4].
            t2_target: Next landing target [B, 4].

        Returns:
            p_com: Dynamically balanced 3D CoM position [B, 3].
            v_com: CoM velocity [B, 3].
            zmp_ref: Reference ZMP on ground plane [B, 2].
        """
        B = self.num_envs
        s = gait_phase.clamp(0.0, 1.0).unsqueeze(-1)  # [B, 1]
        t = s * self.step_time                         # [B, 1] time in seconds

        # ---------------------------------------------------------------------
        # 1. Foothold & ZMP Waypoints (2D XY plane)
        # ---------------------------------------------------------------------
        z_stance = p_stance[:, :2]   # [B, 2] Active stance foothold (ZMP pivot)
        z_t1     = t1_target[:, :2]  # [B, 2] Next foothold
        z_t2     = t2_target[:, :2]  # [B, 2] Upcoming foothold

        # Initial CoM position at start of step (centered between stance & swing start)
        x_0 = 0.5 * (p_stance[:, :2] + p_swing_start[:, :2])  # [B, 2]

        # ---------------------------------------------------------------------
        # 2. Terminal & Initial DCM (Capture Point) Boundary Conditions
        # ---------------------------------------------------------------------
        # Terminal DCM at end of step to ensure smooth continuous transition into next foothold
        # xi_eos = z_t1 + (z_t2 - z_t1) / (exp(omega * T) - 1)
        denom = max(self.exp_omega_T - 1.0, 1e-4)
        xi_eos = z_t1 + (z_t2 - z_t1) / denom  # [B, 2]

        # Initial DCM at start of step: xi_ini = z_stance + (xi_eos - z_stance) * exp(-omega * T)
        xi_ini = z_stance + (xi_eos - z_stance) * (1.0 / self.exp_omega_T)  # [B, 2]

        # ---------------------------------------------------------------------
        # 3. Continuous Analytical DCM & Dynamic CoM Evolution
        # ---------------------------------------------------------------------
        exp_wt   = torch.exp(self.omega * t)           # [B, 1]
        exp_neg_wt = torch.exp(-self.omega * t)         # [B, 1]
        sinh_wt  = 0.5 * (exp_wt - exp_neg_wt)         # [B, 1]
        cosh_wt  = 0.5 * (exp_wt + exp_neg_wt)         # [B, 1]

        # Continuous DCM: xi(t) = z_stance + (xi_ini - z_stance) * exp(omega * t)
        xi_t = z_stance + (xi_ini - z_stance) * exp_wt  # [B, 2]

        # Continuous CoM:
        # x_com(t) = z_stance + (x_0 - z_stance) * exp(-omega * t) + (xi_ini - z_stance) * sinh(omega * t)
        p_com_xy = z_stance + (x_0 - z_stance) * exp_neg_wt + (xi_ini - z_stance) * sinh_wt  # [B, 2]

        # Continuous CoM Velocity:
        # v_com(t) = omega * (xi(t) - x_com(t))
        v_com_xy = self.omega * (xi_t - p_com_xy)  # [B, 2]

        # ---------------------------------------------------------------------
        # 4. Double-Support Phase (DSP) ZMP Smoothing
        # ---------------------------------------------------------------------
        # During initial DSP (s < dsp_ratio), smoothly transfer ZMP from previous support
        dsp_s = (s / max(self.dsp_ratio, 1e-4)).clamp(0.0, 1.0)
        # Cubic smoothstep: 3*s^2 - 2*s^3
        smooth_w = 3.0 * (dsp_s ** 2) - 2.0 * (dsp_s ** 3)
        zmp_ref_xy = (1.0 - smooth_w) * p_swing_start[:, :2] + smooth_w * z_stance  # [B, 2]

        # Assemble full 3D CoM position & velocity with constant / height profile
        z_com = torch.full((B, 1), self.com_height, device=self.device, dtype=torch.float32)
        v_com_z = torch.zeros((B, 1), device=self.device, dtype=torch.float32)

        p_com = torch.cat([p_com_xy, z_com], dim=-1)      # [B, 3]
        v_com = torch.cat([v_com_xy, v_com_z], dim=-1)    # [B, 3]

        return p_com, v_com, zmp_ref_xy
