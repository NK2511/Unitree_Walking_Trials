# Comparative Analysis: LIPM-Guided Footstep Planning (Shin et al., 2026) vs. Angad Footstep Planning Trials

**Reference Paper:** *Learning Footstep Constrained Policies Guided by Linear Inverted Pendulum Model for Humanoid Robots* (Jaeyong Shin, Woohyun Cha, Jaeheung Park — Seoul National University / IEEE Access 2026).  
**Our Workspace:** `Angad_Footstep_Planning_Trials` (Humanoid Project DRDO IITK — Angad 13-DOF Biped).

---

## Executive Summary

Both frameworks address **footstep-constrained locomotion** (directing a humanoid to place its feet on precise spatial targets like stepping stones rather than following high-level $v_x, v_y, \omega_z$ velocity commands). However, they adopt fundamentally different control paradigms:

1. **Shin et al. (The Paper)** uses a **Model-Guided Hierarchical Architecture**: An online Linear Inverted Pendulum Model (LIPM) + ZMP Preview Controller (Kajita et al.) computes continuous Center-of-Mass (CoM) and foot trajectories, feeds them through an online analytical Inverse Kinematics (IK) solver at 125 Hz, and trains a **Torque-level DRL Policy** to act as a learned inverse-dynamics tracking controller.
2. **Angad Footstep Planning Trials (Our Implementation)** uses a **Spatial Target-Conditioned / Model-Free RL Architecture with Kinematic Priors**: An online GPU-native `FootstepManager` generates multi-segment spline paths with invisible 3D spatial targets, passes target-relative observation vectors ($t_1, t_2$) to a **Position-controlled DRL Policy (50 Hz via `mjlab` + `mjwarp`)**, and uses phase-synchronized gait clock rewards (Rohan P. Singh style) combined with dynamic kinematic imitation.

---

## Comprehensive Architecture Comparison

```
=======================================================================================================
               SHIN ET AL. (LIPM-GUIDED DRL)                 |         ANGAD FOOTSTEP PLANNING TRIALS
=======================================================================================================
[High-Level Footstep Planner / ArUco]                        | [Multi-Segment Spline / Interactive Designer]
                 │                                            |                      │
                 ▼                                            |                      ▼
[Foothold Command C = (px, py, yaw, tDSP, tSSP, h)]          | [Spatial 3D Waypoint Sequences (x, y, z, θ)]
                 │                                            |                      │
                 ▼                                            |                      ▼
┌──────────────────────────────────────────────┐              | ┌──────────────────────────────────────────────┐
│ ONLINE MODEL-BASED REFERENCE GENERATOR       │              | │ GPU-NATIVE FOOTSTEP MANAGER                  │
│  1. ZMP Reference Trajectory (DSP/SSP)       │              | │  1. PCHIP Multi-Segment Spline Curves        │
│  2. LIPM + Preview Control (1.6s horizon)    │              | │  2. 2-Axis Curriculum (0.25m -> 0.60m)       │
│     --> Continuous 3D CoM Trajectory         │              | │  3. Relative Body-Frame Targets [t1, t2]     │
│  3. Cubic Foot Trajectory Generator          │              | │  4. Touchdown Debounce & Auto-Advancement    │
│  4. Analytical Whole-Body IK Solver (125 Hz) │              | └──────────────────────┬───────────────────────┘
│     --> Target Joint Angles q_des(t+1)       │                                       │
└──────────────────────┬───────────────────────┘                                       │
                       │                                                               │
                       ▼                                                               ▼
┌──────────────────────────────────────────────┐              ┌──────────────────────────────────────────────┐
│ ACTOR NETWORK (TORQUE POLICY - 125 Hz)       │              │ ACTOR NETWORK (POSITION POLICY - 50 Hz)      │
│  • Obs: [v_b, ω_b, g, q, q_dot, q_des,       │              │  • Obs: [r_rp, ω_b, q, q_dot, a_prev,        │
│          Clock, C1, a_prev] (65-dim, 10-step)│              │          t1(x,y,z,θ), t2(x,y,z,θ), Clock]    │
│  • Action: Joint Torques τ = τ_max * a       │              │  • Action: Joint Pos Offsets Δq (scale=0.25) │
│  • Reward: Dense Tracking of LIPM & Foot Traj│              │  • Reward: Hit Bonus, Progress, Phase Clock  │
└──────────────────────┬───────────────────────┘              └──────────────────────┬───────────────────────┘
                       │                                                               │
                       ▼                                                               ▼
        [Direct Joint Torque Command]                         [Low-Level Joint PD Controller]
                       │                                                               │
                       ▼                                                               ▼
        [IsaacGym / Custom SNU Humanoid]                      [MuJoCo mjwarp / Angad 13-DOF Humanoid]
=======================================================================================================
```

---

## Detailed Comparative Breakdown

### 1. Control Philosophy & Reference Generation

| Feature | Shin et al. (Paper) | Angad Implementation |
|:---|:---|:---|
| **Core Paradigm** | **LIPM-Guided Reference Tracking.** The policy acts as a dynamic tracker for an online analytical model. | **Goal-Conditioned Policy.** The policy learns whole-body stepping behavior directly from spatial target vectors. |
| **CoM Trajectory** | **Explicitly computed online** via LIPM Preview Control (ZMP tracking over 1.6s preview window, constant CoM height $z_c = 0.68\text{ m}$). | **Emergent.** CoM motion is not explicitly prescribed; stabilized via pelvis height, upright, and angular momentum rewards. |
| **Inverse Kinematics** | **Online Analytical IK** runs inside the control loop at 125 Hz to generate full target joint trajectories $q_{\text{des}}(t+1)$. | **Offline / Kinematic Imitation Priors.** Mink IK is used offline to generate reference gaits; RL policy directly outputs joint setpoints. |
| **Foot Trajectory** | Cubic spline in Cartesian space computed on-the-fly from $h_{\text{foot}}, t_{\text{DSP}}, t_{\text{SSP}}$. | Generated through spline interpolation in `FootstepManager`, evaluated via spatial hit radii ($r=0.08\text{ m}$). |

---

### 2. Command Representation & Footstep Manager

| Feature | Shin et al. (Paper) | Angad Implementation |
|:---|:---|:---|
| **Command Format** | $C = [p_x^{\text{cmd}}, p_y^{\text{cmd}}, \gamma_z^{\text{cmd}}, t_{\text{DSP}}, t_{\text{SSP}}, h_{\text{foot}}] \in \mathbb{R}^6$ expressed in the **support foot frame**. | Target sequence $T = [x, y, z, \theta]_{1..N}$ expressed as **next 2 targets in robot pelvis/body frame** ($8\text{D}$). |
| **Command Buffer** | Fixed 2-element buffer $[C_1, C_2]$ representing current and upcoming single footstep. | Full trajectory buffer of 20+ steps generated via PCHIP cubic splines or custom CSV designer. |
| **Target Switching** | Synchronized with fixed cycle clock $t_{\text{total}} = t_{\text{SSP}} + 2 \cdot t_{\text{DSP}}$. | **Event-driven & Debounced.** Advanced when foot enters target radius ($< 8\text{ cm}$) and stays for 3 frames ($60\text{ ms}$). |
| **Path Versatility** | Forward, lateral, and yaw stepping within a constrained feasibility fan. | Full multi-segment path generation (straight, curved left/right, backward, lateral, stairs/height gradients). |
| **Curriculum Strategy** | Fixed command bounds (uniform sampling from feasibility envelope). | **2-Axis Curriculum:** Dynamically scales stride distance from $0.25\text{ m}$ (shuffle) to $0.60\text{ m}$ over iterations 500–5000. |

---

### 3. Action Space & Actuation Level

| Feature | Shin et al. (Paper) | Angad Implementation |
|:---|:---|:---|
| **Action Type** | **Direct Joint Torque:** $\tau = \tau_{\text{max}} \odot a_t, \; a_t \in [-1, 1]^{12}$. | **Joint Position Offsets:** $q_{\text{target}} = \bar{q} + K_{\text{scale}} \odot a_t, \; K_{\text{scale}} = 0.25\text{ rad}$. |
| **Control Frequency** | **125 Hz** (8 ms period). Pure torque control requires high frequency to ensure dynamic stability. | **50 Hz** (20 ms period with 4× physics decimation at 200 Hz / 5 ms physics step in MuJoCo). |
| **Actuation Benefit** | High compliance, natural shock absorption, no manual PD gain tuning needed. | High sample efficiency, stability during early exploration, compatible with standard position-controlled actuators. |

---

### 4. Observation Space

| Observation Element | Shin et al. ($\mathbb{R}^{65}$ + 10-Step History) | Angad Implementation ($\mathbb{R}^{51}$) |
|:---|:---|:---|
| **Base Kinematics** | Linear velocity ($3$), Angular velocity ($3$), Projected gravity ($3$). | Root Roll & Pitch ($2$), Angular velocity ($3$). (Linear velocity excluded to force target-based motion). |
| **Joint States** | Positions $q$ ($12$), Velocities $\dot{q}$ ($12$). | Positions $q$ ($12$), Velocities $\dot{q}$ ($12$). |
| **Target Reference** | **Target joint angles from online IK:** $q_{\text{des}}(t+1)$ ($12$). | **Spatial targets relative to pelvis:** $[t_{1,x}, t_{1,y}, t_{1,z}, t_{1,\theta}, t_{2,x}, t_{2,y}, t_{2,z}, t_{2,\theta}]$ ($8$). |
| **Command Vector** | $C_1 \in \mathbb{R}^6$ (foothold + timing). | Embedded in the relative target observation vectors. |
| **Gait Phase** | 2D clock $[\cos\Phi, \sin\Phi]$ ($2$). | 2D gait clock $[\sin\Phi, \cos\Phi]$ ($2$). |
| **Action History** | Previous torque action $a_{t-1}$ ($12$). | Previous position action $a_{t-1}$ ($12$). |
| **Temporal History** | 10-step observation history with frame skipping. | Single-step Markovian state (history captured implicitly through $a_{t-1}$ and clock). |

---

### 5. Reward Formulation & Loss Functions

```
+------------------------------------------------------------------------------------------------------+
|                                          REWARD COMPARISON                                           |
+-----------------------------------+------------------------------------------------------------------+
| Shin et al. (Tracking Formulation)| Angad Trials (Task & Spatial Formulation)                        |
+-----------------------------------+------------------------------------------------------------------+
| r_total = r_track + r_nrg + r_surv| r_total = Σ w_i * r_i                                            |
|                                   |                                                                  |
| • Dense Tracking Rewards:         | • Primary Objective:                                             |
|   - CoM Position & Orientation    |   - footstep_hit (exp bonus when foot lands inside target radius)|
|   - Support Foot Pos/Ori/Vel      |   - footstep_progress (pulls pelvis toward active target t1)     |
|   - Swing Foot Pos & Orientation  |                                                                  |
|   - Target Joint Angles (q_des)   | • Gait Synchronization & Style:                                  |
|                                   |   - foot_clock (Rohan tan-based stance-force & swing-vel reward) |
| • Regularization:                 |   - dynamic_kinematic_imitation (reference walking prior)        |
|   - Energy: -0.0002 * (τ · q_dot) |                                                                  |
|   - Survival: +0.1                | • Stability & Safety Penalties:                                  |
|                                   |   - pelvis_height, upright, footstep_orient                      |
| • Auxiliary Policy Losses:        |   - foot_clearance, foot_slip, foot_swing_height                 |
|   - Mirror Symmetry Loss (w=4.0)  |   - soft_landing, action_rate_l2, dof_pos_limits                 |
|   - Lipschitz Smoothness (w=0.003)|                                                                  |
+-----------------------------------+------------------------------------------------------------------+
```

---

### 6. Simulation & Sim-to-Real Transfer Strategy

| Aspect | Shin et al. (Paper) | Angad Implementation |
|:---|:---|:---|
| **Physics Engine** | **NVIDIA IsaacGym** (PhysX backend). | **MuJoCo / `mjlab` + `mjwarp`** (NVIDIA Warp CUDA-compiled MuJoCo physics). |
| **Footstep Spawning** | Physical stepping stone boxes with collision geometry. | **Invisible Tensor Targets:** Because `mjwarp` compiles static CUDA kernels, targets exist purely as tensors in observation/reward spaces. |
| **Perturbation Strategy** | **State-dependent joint torque perturbation** (up to $50\text{ Nm}$) + base force pushes ($80\text{ N}$) + action delays ($0\text{--}10\text{ ms}$). | Explicit base push disturbances ($[0, 500\text{ N}]$) + ground friction domain randomization + soft landing penalties. |
| **Real Hardware Validation** | Validated on full-sized SNU humanoid with ArUco marker vision tracking + Unitree G1 simulation transfer. | Designed for Angad 13-DOF biped with parallel pushrod ankle kinematics (`Angad_Ankle_Inverse_Kinematics`) and CAN ID mapping (`Angad_Motor_Signs.csv`). |

---

## Key Takeaways & Potential Synergies for Angad

### Advantages of Our Current Approach:
1. **Computational Simplicity:** No need to run an online numerical/analytical QP preview controller or 12-DOF whole-body IK solver inside every environment step on the GPU.
2. **Arbitrary Multi-Segment Path Following:** `FootstepManager` supports continuous spline curves, lateral side-stepping, backward stepping, and stair height climbs with 2-axis progressive curricula.
3. **Robust Position Control:** Higher sample efficiency and safer execution on standard position/PD-actuated hardware.

### Key Innovations from Shin et al. to Consider Adopting:
1. **LIPM CoM Prior for Heavy Payloads:** For heavy humanoids with high gear reduction, explicitly generating a dynamic CoM trajectory via ZMP preview control drastically reduces vertical Ground Reaction Force (GRF) spikes and eliminates foot slamming.
2. **Torque-Level Perturbation Injection:** Training with state-dependent torque noise ($\pm 50\text{ Nm}$) proved effective for zero-shot sim-to-real transfer without extensive domain randomization sweeps.
3. **Mirror Symmetry & Lipschitz Smoothness Losses:** Adding $L_{\text{mirror}}$ and $L_{\text{smooth}}$ loss terms during PPO updates prevents asymmetrical limping and high-frequency actuator jitter.
