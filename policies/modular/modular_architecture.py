"""Modular Multi-Expert Neural Architecture for Unitree G1 Humanoid.

Approach 1: Decoupled Functional Sub-Networks with Interpretable Gating.

Unlike monolithic black-box MLPs where all 49 sensory inputs and 12 motor outputs
are entangled across a single dense weight matrix, this architecture decomposes
the robot's brain into 3 distinct, physically specialized modules:

1. Balance & Postural Stability Module (Balance Expert)
   - Inputs: Base angular velocity (gyro), projected gravity vector, base linear velocity
   - Specialization: Posture stabilization, tilt recovery, ground inclination adaptation

2. Locomotion & Stepping Kinematics Module (Gait Expert)
   - Inputs: Gait clock [sin(phi), cos(phi)], 12 joint positions, 12 joint velocities
   - Specialization: Cyclical swing/stance trajectories, foot clearance, rhythmic leg motion

3. Command Tracking & Action Continuity Module (Navigation Expert)
   - Inputs: Joystick velocity commands [v_x, v_y, omega_z], previous joint action targets
   - Specialization: Velocity scaling, direction turning, torque smoothness

4. Interpretable Gating Unit:
   - Dynamic Softmax arbitration layer producing attention weights:
     alpha_balance + alpha_gait + alpha_command = 1.0
   - Real-time transparency: Allows developers and evaluators to see EXACTLY
     which sub-brain is driving the robot at every millisecond.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Normal


class ObservationSlicer:
    """Explicit semantic index mapping for the Unitree G1 49-dim observation vector.
    
    Observation layout (49 dimensions):
    - Base Linear Velocity (estimated): 3 dims [0:3]
    - Base Angular Velocity (IMU Gyro):  3 dims [3:6]
    - Projected Gravity Vector:          3 dims [6:9]
    - Target Velocity Command:           3 dims [9:12]
    - Relative Joint Positions:         12 dims [12:24]
    - Relative Joint Velocities:        12 dims [24:36]
    - Previous Actions:                 12 dims [36:48]
    - Gait Phase Clock (sin, cos):       2 dims [48:50] (if 49, clock has 1 or joint_pos has 13)
    """

    def __init__(self, obs_dim: int = 49, num_actions: int = 12):
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        # Flexible slicing supporting 48, 49, or 50 observation configurations
        # Default: 49 dims (3 lin_vel, 3 ang_vel, 3 gravity, 3 cmd, 12 pos, 12 vel, 12 act, 1-2 clock)
        self.idx_lin_vel = slice(0, 3)
        self.idx_ang_vel = slice(3, 6)
        self.idx_gravity = slice(6, 9)
        self.idx_command = slice(9, 12)
        self.idx_joint_pos = slice(12, 24)
        self.idx_joint_vel = slice(24, 36)
        self.idx_prev_actions = slice(36, 48)
        self.idx_gait_clock = slice(48, obs_dim)

    def extract_balance_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract sensors critical for balance: IMU gyro (3), Gravity (3), Linear vel (3) = 9 dims."""
        return torch.cat([
            obs[:, self.idx_lin_vel],
            obs[:, self.idx_ang_vel],
            obs[:, self.idx_gravity]
        ], dim=-1)

    def extract_gait_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract sensors critical for stepping: Joint pos (12), Joint vel (12), Gait clock (1-2) = 25-26 dims."""
        clock = obs[:, self.idx_gait_clock] if obs.shape[-1] > 48 else torch.zeros((obs.shape[0], 2), device=obs.device)
        return torch.cat([
            obs[:, self.idx_joint_pos],
            obs[:, self.idx_joint_vel],
            clock
        ], dim=-1)

    def extract_command_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract sensors critical for user navigation: Commands (3), Prev Actions (12) = 15 dims."""
        return torch.cat([
            obs[:, self.idx_command],
            obs[:, self.idx_prev_actions]
        ], dim=-1)


class BalanceExpert(nn.Module):
    """Sub-network 1: Dedicated exclusively to Posture, Balance, and Upright Stabilization."""

    def __init__(self, in_dim: int = 9, hidden_dims: Tuple[int, ...] = (64, 64), out_dim: int = 12):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ELU())
            curr_dim = h
        self.feature_net = nn.Sequential(*layers)
        self.latent_dim = curr_dim
        # Direct joint offset corrections for balance (torso lean, ankle pitch/roll)
        self.action_head = nn.Linear(curr_dim, out_dim)

    def forward(self, x_balance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.feature_net(x_balance)
        balance_actions = self.action_head(latent)
        return balance_actions, latent


class GaitExpert(nn.Module):
    """Sub-network 2: Dedicated exclusively to Cyclic Stepping, Swing/Stance Kinematics, and Clearance."""

    def __init__(self, in_dim: int = 26, hidden_dims: Tuple[int, ...] = (128, 64), out_dim: int = 12):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ELU())
            curr_dim = h
        self.feature_net = nn.Sequential(*layers)
        self.latent_dim = curr_dim
        # Nominal stepping joint trajectory (knees, hips, ankle swing)
        self.action_head = nn.Linear(curr_dim, out_dim)

    def forward(self, x_gait: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.feature_net(x_gait)
        gait_actions = self.action_head(latent)
        return gait_actions, latent


class CommandExpert(nn.Module):
    """Sub-network 3: Dedicated exclusively to Joystick Velocity Tracking & Continuity."""

    def __init__(self, in_dim: int = 15, hidden_dims: Tuple[int, ...] = (64, 64), out_dim: int = 12):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ELU())
            curr_dim = h
        self.feature_net = nn.Sequential(*layers)
        self.latent_dim = curr_dim
        # Modulation of stride length, turning rate, and velocity scaling
        self.action_head = nn.Linear(curr_dim, out_dim)

    def forward(self, x_cmd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.feature_net(x_cmd)
        cmd_actions = self.action_head(latent)
        return cmd_actions, latent


class InterpretableGatingNetwork(nn.Module):
    """Dynamic Gating Unit that arbitrates between the 3 sub-brains.
    
    Produces attention weights [alpha_balance, alpha_gait, alpha_command]
    summing to 1.0 via Softmax.
    """

    def __init__(self, total_latent_dim: int = 64 + 64 + 64, num_experts: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(total_latent_dim, 32),
            nn.ELU(),
            nn.Linear(32, num_experts)
        )

    def forward(self, combined_latents: torch.Tensor) -> torch.Tensor:
        logits = self.gate(combined_latents)
        weights = torch.softmax(logits, dim=-1)
        return weights  # Shape: (batch_size, 3)


class ModularActor(nn.Module):
    """Complete Modular Actor uniting the 3 functional experts and the gating unit."""

    def __init__(
        self,
        obs_dim: int = 49,
        num_actions: int = 12,
        init_noise_std: float = 0.5,
        balance_hidden: Tuple[int, ...] = (64, 64),
        gait_hidden: Tuple[int, ...] = (128, 64),
        cmd_hidden: Tuple[int, ...] = (64, 64),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.slicer = ObservationSlicer(obs_dim=obs_dim, num_actions=num_actions)

        # 3 Dedicated Functional Sub-Networks
        # 1. Balance inputs: 3 (lin vel) + 3 (ang vel) + 3 (gravity) = 9
        self.balance_expert = BalanceExpert(in_dim=9, hidden_dims=balance_hidden, out_dim=num_actions)
        
        # 2. Gait inputs: 12 (pos) + 12 (vel) + (obs_dim - 48) (clock)
        clock_dim = max(0, obs_dim - 48)
        gait_in_dim = 24 + clock_dim
        self.gait_expert = GaitExpert(in_dim=gait_in_dim, hidden_dims=gait_hidden, out_dim=num_actions)
        
        # 3. Command inputs: 3 (cmd) + 12 (prev_actions) = 15
        self.cmd_expert = CommandExpert(in_dim=15, hidden_dims=cmd_hidden, out_dim=num_actions)

        # Dynamic Gating Network
        total_latent = self.balance_expert.latent_dim + self.gait_expert.latent_dim + self.cmd_expert.latent_dim
        self.gating = InterpretableGatingNetwork(total_latent_dim=total_latent, num_experts=3)

        # Trainable log standard deviation for continuous Gaussian exploration
        self.log_std = nn.Parameter(torch.full((num_actions,), torch.log(torch.tensor(init_noise_std))))

    def forward_experts(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute individual expert outputs and gating weights.
        
        Returns:
            blended_mean_action: (batch_size, 12)
            gating_weights: (batch_size, 3) [alpha_bal, alpha_gait, alpha_cmd]
            diagnostics: dictionary containing raw expert actions and individual latents
        """
        x_bal = self.slicer.extract_balance_inputs(obs)
        x_gait = self.slicer.extract_gait_inputs(obs)
        x_cmd = self.slicer.extract_command_inputs(obs)

        a_bal, h_bal = self.balance_expert(x_bal)
        a_gait, h_gait = self.gait_expert(x_gait)
        a_cmd, h_cmd = self.cmd_expert(x_cmd)

        combined_latents = torch.cat([h_bal, h_gait, h_cmd], dim=-1)
        weights = self.gating(combined_latents)  # (batch_size, 3)

        # Explicit interpretable linear combination
        alpha_bal = weights[:, 0:1]
        alpha_gait = weights[:, 1:2]
        alpha_cmd = weights[:, 2:3]

        mean_action = (alpha_bal * a_bal) + (alpha_gait * a_gait) + (alpha_cmd * a_cmd)

        diagnostics = {
            "action_balance": a_bal,
            "action_gait": a_gait,
            "action_command": a_cmd,
            "alpha_balance": alpha_bal,
            "alpha_gait": alpha_gait,
            "alpha_command": alpha_cmd,
            "gating_weights": weights,
        }
        return mean_action, weights, diagnostics

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning deterministic action mean."""
        mean_action, _, _ = self.forward_experts(obs)
        return mean_action

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from Gaussian distribution and return log_prob."""
        mean_action, weights, _ = self.forward_experts(obs)
        if deterministic:
            return mean_action, torch.zeros_like(mean_action[:, 0]), weights

        std = torch.exp(self.log_std).expand_as(mean_action)
        dist = Normal(mean_action, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, weights


class ModularCritic(nn.Module):
    """Decomposed Multi-Head Critic for Physically Interpretable Value Estimation.
    
    Instead of estimating 1 single arbitrary scalar, the critic decomposes value into:
    1. V_balance:    Reward stream from upright orientation and survival
    2. V_locomotion: Reward stream from gait rhythm and foot clearance
    3. V_tracking:   Reward stream from linear/angular velocity command tracking
    """

    def __init__(self, obs_dim: int = 49, hidden_dims: Tuple[int, ...] = (256, 128)):
        super().__init__()
        self.obs_dim = obs_dim
        layers = []
        curr_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ELU())
            curr_dim = h
        self.shared_trunk = nn.Sequential(*layers)

        # 3 Interpretable Value Heads
        self.head_balance = nn.Linear(curr_dim, 1)
        self.head_locomotion = nn.Linear(curr_dim, 1)
        self.head_tracking = nn.Linear(curr_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return composite scalar value for standard PPO advantage calculation."""
        feat = self.shared_trunk(obs)
        v_bal = self.head_balance(feat)
        v_loco = self.head_locomotion(feat)
        v_track = self.head_tracking(feat)
        total_value = v_bal + v_loco + v_track
        return total_value

    def evaluate_decomposed(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return explicit breakdown of expected future rewards."""
        feat = self.shared_trunk(obs)
        v_bal = self.head_balance(feat)
        v_loco = self.head_locomotion(feat)
        v_track = self.head_tracking(feat)
        return {
            "v_total": v_bal + v_loco + v_track,
            "v_balance": v_bal,
            "v_locomotion": v_loco,
            "v_tracking": v_track,
        }
