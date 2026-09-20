# 🤖 Unitree Master Locomotion Workspace

This directory is a **100% self-contained, independent master workspace** for developing, optimizing, training, evaluating, and deploying locomotion algorithms on **Unitree Humanoid Robots (G1 / H1)**.

It contains both **COT (Cost of Transport) / Froude Gait Optimization Trials** and **Dense Footstep Planning RL Trials**, mirroring the full pipeline developed for bipedal locomotion.

---

## 📂 Master Directory Layout

```text
Unitree/
├── Unitree_Asimov_Trials/             # COT Optimization, Froude Walking & Velocity Tracking
│   ├── Gaits/
│   │   ├── gait_generator_cot_dynamic.py     # Cost of Transport (COT) dynamic gait optimizer
│   │   ├── gait_generator_froude_dynamic.py   # Froude number dynamic walking generator
│   │   ├── gait_generator_single_speed.py    # Single-speed gait generator & dataset builder
│   │   ├── gait_player.py                    # Play and visualize optimized gaits in MuJoCo
│   │   └── README_COT_Optimizer.md           # COT Optimization theory & tuning guide
│   └── Walk/
│       ├── unitree_constants.py              # Unitree G1 motor gains (KP/KD), mass, and XML paths
│       ├── train.py                          # Asimov velocity tracking RL trainer
│       ├── policy_evaluator.py               # Evaluates trained velocity tracking policy
│       └── config/ & mdp/                    # Environment & reward specifications
│
├── Unitree_Footstep_Planning_Trials/  # Dense 3D Footstep Target Planning & Target Tracking RL
│   ├── train.py                          # Batched 4096-env footstep RL policy trainer
│   ├── play.py                           # Interactive MuJoCo GUI viewer with stepping stones
│   ├── plot_policy_diagnostics.py        # Automated policy telemetry & 2D trajectory plotter
│   ├── verify_footstep_geometry.py       # Spawn alignment & 3D stepping stone visualizer
│   ├── config/
│   │   ├── unitree_g1_cfg.py             # Unitree G1 KP/KD gain table & environment config
│   │   └── footstep_env_cfg.py           # Footstep environment rewards & observations
│   └── mdp/
│       ├── footstep_manager.py           # Time-synchronized target step sequencer
│       ├── footstep_rewards.py           # Target hit, LIPM CoM, foot_clock, & orientation rewards
│       ├── kinematic_trajectory.py       # Dynamic 2-link leg arc IK reference generator
│       └── lipm_dcm_generator.py         # Analytical 3D LIPM & DCM (Capture Point) solver
│
├── asimov-mjlab/                         # Bundled core simulation & RL framework engine
└── xmls/unitree_g1/                      # Official Unitree G1 MJCF models, meshes, & scenes
```

---

## ⚡ 1. Running COT (Cost of Transport) & Froude Walking

### A. Dynamic COT Gait Optimization
Generates energetically-optimal walking trajectories by minimizing mechanical Cost of Transport ($\text{COT} = P / (m \cdot g \cdot v)$):

```bash
cd Unitree/Unitree_Asimov_Trials/Gaits
python gait_generator_cot_dynamic.py
```

### B. Froude-Number Scaled Walking Dynamics
Generates dynamically-scaled walking trajectories based on Froude number ($Fr = v / \sqrt{g \cdot L}$):

```bash
cd Unitree/Unitree_Asimov_Trials/Gaits
python gait_generator_froude_dynamic.py
```

### C. Play & Visualize Optimized Gaits
Visualize the optimized gaits in MuJoCo GUI:

```bash
cd Unitree/Unitree_Asimov_Trials/Gaits
python gait_player.py
```

---

## 👣 2. Running Footstep Target Planning RL

### A. Train Policy (4096 parallel environments)

```bash
cd Unitree/Unitree_Footstep_Planning_Trials
python train.py Mjlab-Footstep-Flat-Angad --agent.experiment_name unitree_g1_footstep_v1
```

### B. Evaluate Policy in GUI (`play.py`)

```bash
cd Unitree/Unitree_Footstep_Planning_Trials
python play.py
```

### C. Policy Diagnostics & Telemetry

```bash
cd Unitree/Unitree_Footstep_Planning_Trials
python plot_policy_diagnostics.py
```

---

## ⚙️ Unitree Motor Gains & Hardware Tuning

All Unitree G1 actuator stiffness ($K_p$) and damping ($K_d$) control gains are defined in:
- `Unitree_Asimov_Trials/Walk/unitree_constants.py`
- `Unitree_Footstep_Planning_Trials/config/unitree_g1_cfg.py`

```python
UNITREE_G1_GAINS = {
    # Leg joints (Stiffness N·m/rad, Damping N·m·s/rad)
    "hip_pitch":    {"kp": 100.0, "kd": 2.5, "action_scale": 0.25},
    "hip_roll":     {"kp": 100.0, "kd": 2.5, "action_scale": 0.25},
    "hip_yaw":      {"kp": 100.0, "kd": 2.5, "action_scale": 0.25},
    "knee":         {"kp": 150.0, "kd": 4.0, "action_scale": 0.25},
    "ankle_pitch":  {"kp": 40.0,  "kd": 1.0, "action_scale": 0.25},
    "ankle_roll":   {"kp": 40.0,  "kd": 1.0, "action_scale": 0.25},
}
```

This workspace is **100% self-contained** and can be moved to any directory or system.
