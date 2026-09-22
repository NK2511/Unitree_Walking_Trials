# Unitree G1 Humanoid — Modular Multi-Expert RL Architecture (Approach 1)

## 🎯 Overview & Motivation
Standard deep reinforcement learning policies for bipedal robots rely on a **monolithic multi-layer perceptron (MLP)** (e.g. `256 x 256 x 128`). While effective, these monolithic networks suffer from severe **polysemanticity and entanglement**:
- Every single neuron participates in balance, foot swinging, speed tracking, and torque dampening simultaneously.
- When the robot stumbles or exhibits unnatural gait patterns, it is impossible to isolate whether the failure stems from state estimation, posture control, or gait generation.
- Evaluators and academic panels rightly critique it as an uninterpretable **"black box."**

### 🧠 The Solution: Decoupled Sub-Networks with Dynamic Gating
This package implements **Approach 1**, decomposing the humanoid controller into **3 physically isolated, functionally specialized neural sub-networks**:

```
                                      ┌──► [Balance Expert Sub-Net]  (IMU / Tilt)  ────┐ (action_bal)
[49 Sensory Inputs] ──► [Obs Slicer]  ├──► [Gait & Stepping Sub-Net] (Joints / Phase) ───┼──► [Dynamic Gating Softmax] ──► [12 Joint Actions]
                                      └──► [Command Tracking Sub-Net](Target Vel) ─────┘ (action_cmd)
                                                                                              (α_bal, α_gait, α_cmd)
```

---

## 🏛️ Sub-Network Specialization

| Sub-Network | Input Dimensions | Dedicated Sensors | Functional Competence |
| :--- | :--- | :--- | :--- |
| **1. Balance Expert** | **9 dims** | IMU Gyro (`3`), Projected Gravity (`3`), Base Lin Vel (`3`) | Upright posture, dynamic push recovery, ground slope adaptation |
| **2. Gait & Stepping Expert**| **25 dims** | Joint Positions (`12`), Joint Velocities (`12`), Phase Clock (`1`) | Cyclical swing/stance trajectories, foot clearance, rhythmic stepping |
| **3. Command Tracking Expert**| **15 dims** | Target Velocities $[v_x, v_y, \omega_z]$ (`3`), Previous Actions (`12`) | Joystick tracking, directional steering, torque smoothing |

### Real-Time Interpretable Gating
The gating unit dynamically arbitrates between the sub-brains via Softmax:
$$\alpha_{\text{balance}} + \alpha_{\text{gait}} + \alpha_{\text{command}} = 1.0$$

$$\mathbf{a}_{\text{final}} = \alpha_{\text{bal}} \cdot \mathbf{a}_{\text{bal}} + \alpha_{\text{gait}} \cdot \mathbf{a}_{\text{gait}} + \alpha_{\text{cmd}} \cdot \mathbf{a}_{\text{cmd}}$$

* When walking steadily: $\alpha_{\text{gait}} \approx 50\%$, $\alpha_{\text{cmd}} \approx 35\%$, $\alpha_{\text{bal}} \approx 15\%$.
* When pushed or tripped: $\alpha_{\text{bal}} \to 85\%$, instinctively suppressing forward drive to prioritize survival!

---

## 📁 Package Contents

* [`modular_architecture.py`](./modular_architecture.py): Core PyTorch implementation containing `BalanceExpert`, `GaitExpert`, `CommandExpert`, `InterpretableGatingNetwork`, `ModularActor`, and `ModularCritic`.
* [`rsl_rl_wrapper.py`](./rsl_rl_wrapper.py): Drop-in wrapper fully compatible with the RSL-RL PPO training loops and `OnPolicyRunner`.
* [`config_modular.py`](./config_modular.py): Hyperparameters and network configuration.
* [`test_modular_brain.py`](./test_modular_brain.py): Standalone verification test suite checking tensor dimensions, forward passes, gradient backprop, and ONNX export.
* [`visualize_gating.py`](./visualize_gating.py): Telemetry simulation tool that outputs live attention graphs (`modular_gating_telemetry.png`).
* [`train_modular_ubuntu.py`](./train_modular_ubuntu.py): Ready-to-run training launcher for Ubuntu GPU workstations.

---

## 🚀 How to Train on Ubuntu (Step-by-Step)

When you switch to your Ubuntu workstation, follow these simple steps:

### Step 1: Clone / Copy Folder
Ensure this folder `Unitree_Modular_RL` is placed alongside your simulation environment (`asimov-mjlab` or Isaac Gym).

### Step 2: Install Dependencies
```bash
conda activate unitree_env
pip install torch torchvision
pip install rsl-rl
pip install -e ../asimov-mjlab
```

### Step 3: Run Sanity Verification Test
```bash
python test_modular_brain.py
```
*(All 6 sanity checks will pass, confirming your environment and dimensions are ready).*

### Step 4: Launch GPU Training
```bash
# Trains with 4,096 parallel environments on your NVIDIA GPU:
python train_modular_ubuntu.py --task Mjlab-Velocity-Flat-Unitree --num_envs 4096 --headless
```

### Step 5: Monitor Live Training via TensorBoard
```bash
tensorboard --logdir ./logs/modular_unitree
```

---

## 📊 How to Present This in Your PPT / Defense

1. **Contrast with Black Box:** Show the previous monolithic weight matrix vs. this modular diagram. Explain that standard RL entangles all behaviors, making safety verification impossible.
2. **Explain Bio-Inspiration:** Explain that in biological locomotion, the spinal central pattern generator handles stepping, vestibular reflexes handle balance, and the motor cortex handles conscious navigation. Approach 1 replicates this exact division of labor.
3. **Show the Attention Graph:** Include `modular_gating_telemetry.png` to demonstrate that you can query the robot's brain in real-time and observe its decision-making shift from walking to recovery during disturbances.
