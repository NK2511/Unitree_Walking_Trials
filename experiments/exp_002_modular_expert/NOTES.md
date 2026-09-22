# Experiment 002 — Modular Multi-Expert Policy

## Overview
Decoupled 3-expert architecture where Balance, Gait, and Command sub-networks
are physically isolated with a dynamic Softmax gating unit for real-time interpretability.

## Architecture
- **Balance Expert**: `MLP (9 → 64 → 64 → 12)` — IMU + gravity inputs only
- **Gait Expert**: `MLP (25 → 128 → 64 → 12)` — joints + phase clock only
- **Command Expert**: `MLP (15 → 64 → 64 → 12)` — joystick + prev actions only
- **Gating Network**: `MLP (192 → 32 → 3)` — dynamic softmax arbitration
- Total actor parameters: **30,195** (6x lighter than baseline)

See `policies/modular/` for implementation.

## Status
- [x] Architecture implemented and verified (test_modular_brain.py: all 6 checks pass)
- [x] Telemetry visualization working (visualize_gating.py)
- [ ] GPU training — run on Ubuntu workstation
- [ ] Evaluation vs exp_001 baseline

## To Train on Ubuntu
```bash
cd policies/modular
python train_modular_ubuntu.py --task Mjlab-Velocity-Flat-Unitree --num_envs 4096 --headless
tensorboard --logdir ./logs/modular_unitree
```

## Hypotheses to Test
1. Will gating weights reliably shift toward Balance Expert during perturbations?
2. Will the 6x parameter reduction hurt sample efficiency or final performance?
3. Can we use gating weight curves as a diagnostic for gait failure modes?
