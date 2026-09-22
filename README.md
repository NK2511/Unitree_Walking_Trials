# Unitree G1 Humanoid — PhD Research Repository

## What This Repo Is
Long-term PhD research workspace for training and studying interpretable reinforcement learning controllers for the **Unitree G1 humanoid robot**.

Core simulation framework: [MJLab](asimov-mjlab/) built on MuJoCo + GPU-accelerated MuJoCo Warp.

---

## How to Find Things

| You want to... | Go to |
| :--- | :--- |
| **Change / swap the neural network brain** | `policies/` (`base_policy.py`, `mlp_policy.py`, `modular/`) |
| **Add or modify a reward function** | `envs/unitree_walk/mdp/rewards.py` (Single source of truth) |
| **Change reward weights** | `envs/unitree_walk/env_cfgs.py` |
| **Change observation space** | `envs/unitree_walk/mdp/observations.py` |
| **Change env params** (num_envs, terrain, sim) | `envs/unitree_walk/env_cfgs.py` |
| **Train the walking controller** | `train.py` (Run from repo root) |
| **See all experiments & research notes** | `experiments/` (`exp_001`, `exp_002`, etc.) |
| **Analyze neural network weights or activations** | `analysis/` |
| **Run interactive joystick teleoperation** | `tools/joystick.py` |
| **See robot hardware / MJCF / constants** | `robots/unitree_g1/` (`constants.py`, `xmls/`) |
| **Simulation framework source** | `asimov-mjlab/src/mjlab/` |

---

## Repository Structure

```
Unitree_Walking_Trials/
├── train.py                     Unified training launcher (python train.py Mjlab-Velocity-Flat-Unitree)
│
├── envs/                        Single source of truth for environments & MDP
│   └── unitree_walk/
│       ├── __init__.py          Task registration (register_mjlab_task)
│       ├── env_cfgs.py          Scene, sim, domain rand, reward weights
│       ├── rl_cfg.py            Baseline PPO runner hyperparameters
│       └── mdp/                 Rewards, observations, curriculums, terminations
│
├── policies/                    Swappable brain architectures
│   ├── base_policy.py           Abstract interface (implement to add new brain)
│   ├── mlp_policy.py            Standard MLP baseline (exp_001)
│   └── modular/                 3-Expert modular architecture (exp_002)
│
├── experiments/                 One folder per research direction
│   ├── exp_001_baseline_walk/   Trained MLP baseline + checkpoints/ (model_5499.pt)
│   ├── exp_002_modular_expert/  Modular architecture training + NOTES.md
│   └── exp_footstep_planning/   Footstep planner track + NOTES.md
│
├── analysis/                    Visualization & evaluation tools
│   ├── neural_network_visualizer.html  Interactive weight/activation viewer
│   ├── generate_brain_weight_images.py
│   ├── policy_evaluator.py
│   ├── batch_eval_graphs.py
│   ├── policy_video_renderer.py
│   └── outputs/                 Generated PNGs, MP4s (gitignored)
│
├── robots/                      Robot hardware definitions
│   └── unitree_g1/
│       ├── constants.py         Joint limits, PD gains, action scales
│       └── xmls/                MJCF model files
│
├── tools/                       Utility & teleoperation tools
│   ├── joystick.py              Interactive keyboard teleoperation
│   ├── update_all_slides.py
│   └── view_g1.py
│
├── Unitree_Asimov_Trials/       Legacy folder (maintained for 100% backward compatibility)
│   └── Walk/                    Transparently forwards to envs/unitree_walk
│
├── docs/                        Research documents
│   ├── literature_survey_poster.md
│   ├── walk_training_cheatsheet.md
│   └── presentations/
│
└── asimov-mjlab/                Simulation framework (upstream)
```


---

## Quickstart

### Run the trained baseline policy (interactive teleoperation)
```bash
cd Unitree_Asimov_Trials/Walk
python joystick.py
```
Arrow keys to control velocity. `X` to stop.

### Train the baseline MLP policy
```bash
cd Unitree_Asimov_Trials/Walk
python train.py
```

### Train the modular expert policy (Ubuntu GPU required)
```bash
cd policies/modular
python train_modular_ubuntu.py --task Mjlab-Velocity-Flat-Unitree --num_envs 4096 --headless
```

### Run neural network weight analysis
```bash
cd analysis
python generate_brain_weight_images.py
# Open analysis/neural_network_visualizer.html in browser
```

---

## Swapping the Policy Brain

All policies implement `policies/base_policy.py`. To use a different architecture:

```python
# In your experiment config or train.py:
from policies.mlp_policy import MlpPolicy           # Experiment 001
from policies.modular import RslRlModularActorCritic  # Experiment 002
```

---

## Current Best Checkpoint
`Unitree_Asimov_Trials/Walk/logs/rsl_rl/unitree_rigid_upper_body/2026-09-20_23-07-18/model_5499.pt`
