# Experiment 001 — Baseline MLP Walk

## Overview
Standard monolithic 3-layer MLP policy trained with PPO on the Unitree G1
flat terrain velocity tracking task.

## Architecture
- Actor: `MLP (49 → 256 → 256 → 128 → 12)` with ELU activations
- Critic: `MLP (49 → 256 → 256 → 128 → 1)` with ELU activations
- Noise std: 0.5 (adaptive schedule)
- Total parameters: ~170,000

## Training Details
| Parameter | Value |
| :--- | :--- |
| Task | `Mjlab-Velocity-Flat-Unitree` |
| Parallel envs | 4096 |
| Max iterations | 5499 (checkpoint) |
| Learning rate | 1e-3 (adaptive KL) |
| Gamma | 0.99, Lambda | 0.95 |

## Best Checkpoint
`experiments/exp_001_baseline_walk/checkpoints/model_5499.pt`
*(Legacy copy: `Unitree_Asimov_Trials/Walk/logs/rsl_rl/unitree_rigid_upper_body/2026-09-20_23-07-18/model_5499.pt`)*

## Observations
- Robot walks stably on flat terrain at commanded velocities up to ~1.5 m/s
- Upper body compensation active but stiff-looking at slow speeds
- No interpretability — cannot diagnose why specific gait failures occur
- Neural network weight analysis shows heavy investment in joint position (31.4) and IMU gyro (27.4)

## What to Try Next
- Reduce upper-body reward weight to allow more natural arm swing
- Increase gait clock reward to enforce sharper rhythm
- See `exp_002_modular_expert` for the interpretable architecture follow-up
