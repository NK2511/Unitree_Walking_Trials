# Unitree G1 Kinematic & Dynamic Gaits

This directory contains the pipeline and tools for generating kinematic walking trajectories (gaits) for the **Unitree G1 Bipedal Robot**. These gaits serve as the reference motions for imitation learning in the RL training pipeline.

---

## 📐 Unitree G1 Biomechanical & Dynamic Parameters

| Property | Value | Description |
| :--- | :---: | :--- |
| **Leg Length ($L_{\text{leg}}$)** | $0.74\text{ m}$ | Distance from hip-pitch axis to foot sole |
| **Total Robot Mass ($M$)** | $35.0\text{ kg}$ | Total G1 mass including actuators & battery |
| **Natural Pendulum Frequency ($f_0$)** | $0.579\text{ Hz}$ | Swing frequency $f_0 = \frac{1}{2\pi}\sqrt{\frac{g}{L_{\text{leg}}}}$ |
| **Walk-to-Run Transition ($V_{\text{TRANSITION}}$)** | $1.906\text{ m/s}$ | Speed corresponding to Froude limit $Fr = 0.5$ |
| **12-DOF Leg Joints** | `left_hip_pitch_joint`<br>`left_hip_roll_joint`<br>`left_hip_yaw_joint`<br>`left_knee_joint`<br>`left_ankle_pitch_joint`<br>`left_ankle_roll_joint` | Symmetric for right leg |

---

## 🛠️ Gait Generator Tools

### 1. Dynamic Cost of Transport (COT) Gait Optimizer
`gait_generator_cot_dynamic.py`
Optimizes stride length ($L$), frequency ($F$), and duty factor ($S$) to minimize mechanical work and electrical Joule heating losses across speeds $0.1\text{--}2.5\text{ m/s}$.

```bash
python gait_generator_cot_dynamic.py
```

### 2. Froude-Scaled Dynamic Gait Generator
`gait_generator_froude_dynamic.py`
Generates dynamically-scaled walking trajectories enforcing Froude similarity rules ($Fr = v^2 / g \cdot L_{\text{leg}}$).

```bash
python gait_generator_froude_dynamic.py
```

### 3. Single-Speed Interactive Gait Designer
`gait_generator_single_speed.py`
Interactive GUI allowing manual waypoint placement and trajectory fine-tuning for specific target speeds.

```bash
python gait_generator_single_speed.py
```

### 4. Interactive Gait Player
`gait_player.py`
Plays and visualizes generated `.csv` / `.npz` gait trajectories in MuJoCo viewer:

```bash
python gait_player.py
```
