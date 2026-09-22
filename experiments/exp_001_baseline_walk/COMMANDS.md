# Angad Walker RL — Command Cheatsheet

Quick reference for training, playing, and evaluating the Angad walking policy.

---

## 1. Activate Environment

```bash
conda activate mjlab_env
```

---

## 2. Navigate to Walker Directory

```bash
cd /home/nandhith/Python/Humanoid_Xterra_IITK/Angad_Asimov_Trials/Walk
```

---

## 3. Train the Policy

### Flat terrain (recommended for initial training):
```bash
python train.py Mjlab-Velocity-Flat-Angad --env.scene.num-envs 4096
```

### Rough terrain (curriculum, run after flat is stable):
```bash
python train.py Mjlab-Velocity-Rough-Angad --env.scene.num-envs 4096
```

### Resume from a specific checkpoint:
```bash
python train.py Mjlab-Velocity-Flat-Angad --env.scene.num-envs 4096 \
  --checkpoint logs/rsl_rl/angad_velocity/<run_folder>/model_XXXXX.pt
```

---

## 4. Play Trained Policy (Joystick)

### Auto-detect latest checkpoint:
```bash
python joystick.py Mjlab-Velocity-Flat-Angad --viewer native
```

### Specify a checkpoint explicitly:
```bash
python joystick.py Mjlab-Velocity-Flat-Angad --viewer native \
  --checkpoint-file logs/rsl_rl/angad_velocity/<run_folder>/model_XXXXX.pt
```

**Keyboard controls (native viewer):**
- Arrow keys: forward / backward / strafe
- Q / E: rotate left / right

---

## 5. Evaluate a Trained Policy

Runs quantitative gait metrics (velocity tracking, foot impact, joint torques) and saves plots:

```bash
cd /home/nandhith/Python/Humanoid_Xterra_IITK/Angad_Asimov_Trials/Gaits

# With 3D viewer:
python Angad_Gait_Evaluator.py ../Walk/logs/rsl_rl/angad_velocity/<run_folder>

# Headless (fast, just plots):
python Angad_Gait_Evaluator.py ../Walk/logs/rsl_rl/angad_velocity/<run_folder> --no-render
```

---

## 6. Gait Generation

### Regenerate the speed-adaptive NPZ (required after changing IK parameters):
```bash
cd /home/nandhith/Python/Humanoid_Xterra_IITK/Angad_Asimov_Trials/Gaits
python Angad_Speed_Gait_Mapper.py
# Close the viewer after "✅ Saved 12-DOF dynamic anchors" appears
```

### Visualize any gait CSV:
```bash
python Play_Gait_CSV.py csvs/2D_Kinematic_Angad_Gait.csv
python Play_Gait_CSV.py csvs/2D_Kinematic_Angad_Gait_With_Sway.csv
python Play_Gait_CSV.py csvs/fundamental_frequency_base_gait.csv
```

---

## 7. Utilities

```bash
cd /home/nandhith/Python/Humanoid_Xterra_IITK/Angad_Asimov_Trials/Utilities

# View any keyframe from the XML:
python Angad_Inspect_Keyframe.py

# Interactive IK stance controller:
python Angad_Interactive_Stance_IK.py

# PD gain tuner:
python Angad_PD_Tuner.py
```

---

## Notes

- **Joint convention:** All CSVs and the NPZ store **raw Angad XML qpos** (no sign flips).
- **NPZ location:** `Angad_Asimov_Trials/Gaits/angad_dynamic_gaits.npz` — this is the canonical file used by training.
- **Pending cleanup:** After regenerating the NPZ, remove the two `sign_flips` TODO blocks in `Walk/mdp/rewards.py`.
- **Logs:** Training checkpoints are in `Walk/logs/rsl_rl/angad_velocity/`. Each run creates a timestamped subdirectory.
