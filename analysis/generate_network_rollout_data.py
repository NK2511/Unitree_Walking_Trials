import os
import sys
import json
from pathlib import Path
import mujoco
import torch
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def generate_network_data():
    print("🧠 Extracting Neural Network Weights & Rollout Activations...")

    ckpt_path = Path("Unitree_Asimov_Trials/Walk/logs/rsl_rl/unitree_rigid_upper_body/2026-09-20_23-07-18/model_5499.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["actor_state_dict"]

    mean = sd["obs_normalizer._mean"].numpy()[0]
    std = sd["obs_normalizer._std"].numpy()[0]
    w0 = sd["mlp.0.weight"].numpy() # (256, 49)
    b0 = sd["mlp.0.bias"].numpy()   # (256,)
    w2 = sd["mlp.2.weight"].numpy() # (256, 256)
    b2 = sd["mlp.2.bias"].numpy()   # (256,)
    w4 = sd["mlp.4.weight"].numpy() # (128, 256)
    b4 = sd["mlp.4.bias"].numpy()   # (128,)
    w6 = sd["mlp.6.weight"].numpy() # (12, 128)
    b6 = sd["mlp.6.bias"].numpy()   # (12,)

    def elu(x): return np.where(x > 0, x, np.exp(x) - 1.0)

    # 1. Extract top 150 strongest synaptic connections per layer transition for sleek visualization
    def get_top_synapses(W, top_k=120):
        flat_idx = np.argsort(np.abs(W).ravel())[-top_k:]
        synapses = []
        for idx in flat_idx:
            out_i, in_j = np.unravel_index(idx, W.shape)
            synapses.append([int(in_j), int(out_i), float(round(float(W[out_i, in_j]), 3))])
        return synapses

    synapses_0_1 = get_top_synapses(w0, top_k=150)
    synapses_1_2 = get_top_synapses(w2, top_k=180)
    synapses_2_3 = get_top_synapses(w4, top_k=150)
    synapses_3_4 = get_top_synapses(w6, top_k=80)

    # 2. Run simulation and record activations across 250 steps (5.0s @ 50Hz, two complete gait cycles)
    spec = mujoco.MjSpec.from_file(r"xmls/unitree_g1/scene_mjx.xml")
    m = spec.compile()
    m.opt.timestep = 0.005
    d = mujoco.MjData(m)

    default_q = np.array([
        -0.2203, 0.0, 0.0, 0.5706, -0.3504, 0.0,
        -0.2203, 0.0, 0.0, 0.5706, -0.3504, 0.0,
        0.0
    ])

    kp = np.array([100., 100., 100., 150., 40., 40., 100., 100., 100., 150., 40., 40., 100.])
    kd = np.array([2.5, 2.5, 2.5, 4.0, 1.0, 1.0, 2.5, 2.5, 2.5, 4.0, 1.0, 1.0, 2.5])
    for i in range(13):
        m.actuator_gainprm[i, 0] = kp[i]
        m.actuator_biasprm[i, 1] = -kp[i]
        m.actuator_biasprm[i, 2] = -kd[i]

    d.qpos[0:3] = [0.0, 0.0, 0.7645]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:20] = default_q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    def quat_apply_inverse(q, v):
        w, x, y, z = q
        qc = np.array([w, -x, -y, -z])
        qv = np.array([0, v[0], v[1], v[2]])
        t0 = qc[0]*qv[0] - qc[1]*qv[1] - qc[2]*qv[2] - qc[3]*qv[3]
        t1 = qc[0]*qv[1] + qc[1]*qv[0] + qc[2]*qv[3] - qc[3]*qv[2]
        t2 = qc[0]*qv[2] - qc[1]*qv[3] + qc[2]*qv[0] + qc[3]*qv[1]
        t3 = qc[0]*qv[3] + qc[1]*qv[2] - qc[2]*qv[1] + qc[3]*qv[0]
        res1 = t0*q[1] + t1*q[0] + t2*q[3] - t3*q[2]
        res2 = t0*q[2] - t1*q[3] + t2*q[0] + t3*q[1]
        res3 = t0*q[3] + t1*q[2] - t2*q[1] + t3*q[0]
        return np.array([res1, res2, res3])

    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro_pelvis")
    g_adr = m.sensor_adr[gid]

    frames = []
    actions = np.zeros(12)
    gait_phase = 0.0
    dt = 0.02
    num_steps = 250 # 5.0 seconds at 50Hz

    print(f"🎬 Simulating {num_steps} steps and recording full-layer activations...")

    for step in range(num_steps):
        t_sim = step * dt
        vx_cmd = 0.40 + (t_sim / 5.0) * 0.60 # 0.40 to 1.00 m/s

        base_ang_vel = d.sensordata[g_adr:g_adr+3]
        proj_grav = quat_apply_inverse(d.qpos[3:7], [0., 0., -1.])
        cmd = np.array([vx_cmd, 0.0, 0.0])
        joint_pos = d.qpos[7:20] - default_q
        joint_vel = d.qvel[6:19]
        clock = np.array([np.cos(2*np.pi*gait_phase), np.sin(2*np.pi*gait_phase)])

        obs = np.concatenate([base_ang_vel, proj_grav, cmd, joint_pos, joint_vel, actions, clock])
        obs_norm = (obs - mean) / std

        # Forward pass activations
        z1 = obs_norm @ w0.T + b0
        a1 = elu(z1)

        z2 = a1 @ w2.T + b2
        a2 = elu(z2)

        z3 = a2 @ w4.T + b4
        a3 = elu(z3)

        z4 = a3 @ w6.T + b6
        actions = z4

        target_q = default_q.copy()
        target_q[:12] += 0.25 * actions
        d.ctrl[:13] = target_q

        prev_x = d.qpos[0]
        for _ in range(4):
            mujoco.mj_step(m, d)
        actual_vx = (d.qpos[0] - prev_x) / dt

        gait_phase = (gait_phase + dt * 1.25) % 1.0

        # Store compact rounded activations
        frame_data = {
            "t": round(t_sim, 2),
            "step": step,
            "cmd_vx": round(float(vx_cmd), 2),
            "actual_vx": round(float(actual_vx), 2),
            "distance": round(float(d.qpos[0]), 2),
            "pelvis_z": round(float(d.qpos[2]), 3),
            "gait_phase": round(float(gait_phase), 2),
            "L0": [round(float(v), 2) for v in obs_norm],
            "L1": [round(float(v), 2) for v in a1],
            "L2": [round(float(v), 2) for v in a2],
            "L3": [round(float(v), 2) for v in a3],
            "L4": [round(float(v), 2) for v in actions],
        }
        frames.append(frame_data)

    input_names = [
        "IMU Gyro X", "IMU Gyro Y", "IMU Gyro Z",
        "Proj Grav X", "Proj Grav Y", "Proj Grav Z",
        "Cmd Vx", "Cmd Vy", "Cmd Yaw",
        "L Hip Pitch Pos", "L Hip Roll Pos", "L Hip Yaw Pos", "L Knee Pos", "L Ankle Pitch Pos", "L Ankle Roll Pos",
        "R Hip Pitch Pos", "R Hip Roll Pos", "R Hip Yaw Pos", "R Knee Pos", "R Ankle Pitch Pos", "R Ankle Roll Pos",
        "Waist Yaw Pos",
        "L Hip Pitch Vel", "L Hip Roll Vel", "L Hip Yaw Vel", "L Knee Vel", "L Ankle Pitch Vel", "L Ankle Roll Vel",
        "R Hip Pitch Vel", "R Hip Roll Vel", "R Hip Yaw Vel", "R Knee Vel", "R Ankle Pitch Vel", "R Ankle Roll Vel",
        "Waist Yaw Vel",
        "Last L Hip Pitch Act", "Last L Hip Roll Act", "Last L Hip Yaw Act", "Last L Knee Act", "Last L Ankle Pitch Act", "Last L Ankle Roll Act",
        "Last R Hip Pitch Act", "Last R Hip Roll Act", "Last R Hip Yaw Act", "Last R Knee Act", "Last R Ankle Pitch Act", "Last R Ankle Roll Act",
        "Gait Clock Cos", "Gait Clock Sin"
    ]

    output_names = [
        "L Hip Pitch Target", "L Hip Roll Target", "L Hip Yaw Target",
        "L Knee Target", "L Ankle Pitch Target", "L Ankle Roll Target",
        "R Hip Pitch Target", "R Hip Roll Target", "R Hip Yaw Target",
        "R Knee Target", "R Ankle Pitch Target", "R Ankle Roll Target"
    ]

    export_data = {
        "architecture": {
            "layers": [49, 256, 256, 128, 12],
            "input_names": input_names,
            "output_names": output_names,
            "synapses": {
                "s01": synapses_0_1,
                "s12": synapses_1_2,
                "s23": synapses_2_3,
                "s34": synapses_3_4
            }
        },
        "frames": frames
    }

    out_file = Path("neural_network_data.json")
    with open(out_file, "w") as f:
        json.dump(export_data, f)

    print(f"✅ Saved neural network rollout data to {out_file.name} ({out_file.stat().st_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    generate_network_data()
