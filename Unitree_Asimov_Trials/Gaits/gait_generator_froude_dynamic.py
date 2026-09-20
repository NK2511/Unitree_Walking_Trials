#!/usr/bin/env python3
"""Angad Speed-to-Gait Mapping Tool.

This script implements speed-to-gait mapping based on Froude scaling rules
and duty factor curves for the Angad robot.
"""

import os
import sys
import csv
import time
import threading
import numpy as np
import argparse

# Auto-relaunch inside the correct virtual environment if not already using it
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.abspath(os.path.join(CURRENT_DIR, "../../mjlab_env/bin/python"))
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    print(f"🔄 Auto-switching to mjlab_env Python interpreter...")
    os.execl(VENV_PYTHON, VENV_PYTHON, *sys.argv)

import re
import mujoco
import mujoco.viewer
import mink
from scipy.interpolate import PchipInterpolator
import tkinter as tk
from tkinter import messagebox

# Configuration
XML_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../xmls/unitree_g1/scene.xml"))
if not os.path.exists(XML_PATH):
    XML_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../xmls/unitree_g1/g1.xml"))

OUT_CSV = os.path.abspath(os.path.join(CURRENT_DIR, "csvs/unitree_g1_speed_mapped_gait.csv"))
KEYFRAME_INDEX = 0
N_FRAMES = 120

# Biomechanics & Measured Parameters for Unitree G1 (Leg Length = 0.74m)
leg_length = 0.74        # m (Unitree G1 leg length)
f0 = 0.579               # Hz (Natural pendulum frequency f0 = 1/(2*pi) * sqrt(g/L))
v_base = 0.15            # m/s
L_base = 0.25            # m
F_base = 0.60            # Hz
S_base = 0.60            # baseline stance duty factor
S_floor = 0.45           # minimum duty factor
a = 0.6                  # scaling exponent
v_transition = 1.906     # walk -> run transition speed (Froude = v^2 / (g*L) = 0.5)

# Joint map: Unitree G1 12-DOF leg joint names
JMAP = {
    "left_hip_pitch_joint":    1.0,
    "left_hip_roll_joint":     1.0,
    "left_hip_yaw_joint":      1.0,
    "left_knee_joint":         1.0,
    "left_ankle_pitch_joint":  1.0,
    "left_ankle_roll_joint":   1.0,
    "right_hip_pitch_joint":   1.0,
    "right_hip_roll_joint":    1.0,
    "right_hip_yaw_joint":     1.0,
    "right_knee_joint":        1.0,
    "right_ankle_pitch_joint": 1.0,
    "right_ankle_roll_joint":  1.0,
}

# Global State
model = data = None
rf_site_id = lf_site_id = None
stance_qpos = rf_stance_world = lf_stance_world = None
waypoints = []

# Biomechanical Scaling Functions
def stride_length(v, L_base=L_base, v_base=v_base):
    # Scale linearly to prevent early saturation, maxing out at 0.58m at 2.5 m/s
    t = max(0.0, min(1.0, (v - v_base) / (2.5 - v_base)))
    L = L_base + (0.58 - L_base) * t
    return L


def stride_frequency(v, L_base=L_base, v_base=v_base):
    L = stride_length(v, L_base, v_base)
    return v / L

def duty_factor(v, v_base=v_base, v_transition=v_transition, S_base=S_base, S_floor=S_floor):
    t = max(0.0, min(1.0, (v - v_base) / (v_transition - v_base)))
    return S_base + (S_floor - S_base) * t

def get_waypoints_for_speed(v, rf_stance):
    L = stride_length(v)
    
    # WP1 (rear / lift-off): x = -L/2
    wp1_x = -L / 2.0
    wp1_pos = (rf_stance[0] + wp1_x, rf_stance[1], rf_stance[2])
    
    # WP2 (swing apex): x = +0.03 * (L / L_base), z = +0.045
    wp2_x = 0.03 * (L / L_base)
    wp2_pos = (rf_stance[0] + wp2_x, rf_stance[1], rf_stance[2] + 0.045)
    
    # WP3 (front / touch-down): x = +L/2
    wp3_x = L / 2.0
    wp3_pos = (rf_stance[0] + wp3_x, rf_stance[1], rf_stance[2])
    
    return [
        {"pos": wp1_pos, "weight": 1.0},
        {"pos": wp2_pos, "weight": 1.0},
        {"pos": wp3_pos, "weight": 1.0}
    ]

# Spline & Timing Helpers (from Angad_Create_Gait.py)
def build_spline_spatial(wp_list):
    n   = len(wp_list)
    pts = np.array([w["pos"] for w in wp_list], dtype=float)
    gnd = rf_stance_world[2]
    pts[0, 2]  = gnd
    pts[-1, 2] = gnd

    pc  = np.vstack([pts, rf_stance_world, pts[0]])
    t_u = np.linspace(0, 1, len(pc))

    return (PchipInterpolator(t_u, pc[:,0]),
            PchipInterpolator(t_u, pc[:,1]),
            PchipInterpolator(t_u, pc[:,2]),
            t_u)

def cycloidal_warp(tau):
    return tau - np.sin(2*np.pi*tau) / (2*np.pi)

def temporal_t_map(speeds, stance_speed, n_frames):
    swing_speed = speeds[0]
    n = len(speeds)
    w_swing = (n - 1) / max(0.1, swing_speed)
    w_stance = 2.0 / max(0.1, stance_speed)

    cum_frac = [0.0, w_swing / (w_swing + w_stance), 1.0]
    t_bounds = [0.0, (n - 1) / (n + 1), 1.0]

    tau = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    p   = cycloidal_warp(tau)

    return np.interp(p, cum_frac, t_bounds)

def solve_ik_traj(speeds, stance_speed, phase_frac, hip_sway=0.03):
    cs_x, cs_y, cs_z, _ = build_spline_spatial(waypoints)
    t  = temporal_t_map(speeds, stance_speed, N_FRAMES)
    tx, ty, tz = cs_x(t), cs_y(t), cs_z(t)
    
    # Force stance phase (t >= 0.5) to have a constant horizontal velocity and perfectly flat Z
    stance_mask = t >= 0.5
    frac = 2.0 * (t[stance_mask] - 0.5)  # 0.0 at t=0.5, 1.0 at t=1.0
    wp3_x = waypoints[2]["pos"][0]
    wp1_x = waypoints[0]["pos"][0]
    tx[stance_mask] = wp3_x * (1.0 - frac) + wp1_x * frac
    tz[stance_mask] = rf_stance_world[2]
    
    tz = np.maximum(tz, rf_stance_world[2])

    d_tmp = mujoco.MjData(model)
    d_tmp.qpos[:] = stance_qpos
    mujoco.mj_forward(model, d_tmp)
    rf_rot = d_tmp.site_xmat[rf_site_id].reshape(3,3).copy()
    lf_rot = d_tmp.site_xmat[lf_site_id].reshape(3,3).copy()
    lf_st  = lf_stance_world.copy()
    pb_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    T_pel  = np.eye(4)
    T_pel[:3,:3] = d_tmp.xmat[pb_id].reshape(3,3)
    T_pel[:3,3]  = d_tmp.xpos[pb_id]
    lf_dx = tx - rf_stance_world[0]
    lf_dz = tz - rf_stance_world[2]
    half = int(round(phase_frac*N_FRAMES))

    cfg = mink.Configuration(model)
    cfg.update(stance_qpos)
    rf_name = "right_foot" if mujoco.mj_name2id(m_tmp, mujoco.mjtObj.mjOBJ_SITE, "right_foot") != -1 else "right_foot_site"
    lf_name = "left_foot" if mujoco.mj_name2id(m_tmp, mujoco.mjtObj.mjOBJ_SITE, "left_foot") != -1 else "left_foot_site"
    rf_t = mink.FrameTask(rf_name, "site", position_cost=2000., orientation_cost=200., lm_damping=1e-2)
    lf_t = mink.FrameTask(lf_name, "site", position_cost=2000., orientation_cost=200., lm_damping=1e-2)
    pel_t = mink.FrameTask("base", "body", position_cost=2000., orientation_cost=2000., lm_damping=1e-2)
    pel_t.set_target(mink.SE3.from_matrix(T_pel))
    post_t = mink.PostureTask(model, cost=1e-4)
    post_t.set_target(stance_qpos)
    tasks = [rf_t, lf_t, pel_t, post_t]
    limits = [mink.ConfigurationLimit(model)]
    DT, MAX_IT, TOL = 0.002, 100, 8e-4
    results = []
    
    # Calculate target pelvis height dynamically based on stride length to avoid leg over-extension
    L = waypoints[2]["pos"][0] - waypoints[0]["pos"][0]
    max_reach = 0.78  # Safe maximum leg reach (nominal leg_length is 0.86) to preserve ankle range of motion
    target_z = min(T_pel[2, 3], np.sqrt(max_reach**2 - (L / 2.0)**2))
    
    for idx in range(N_FRAMES):
        T_rf = np.eye(4)
        T_rf[:3,:3] = rf_rot
        T_rf[:3,3]  = [tx[idx], ty[idx], tz[idx]]
        rf_t.set_target(mink.SE3.from_matrix(T_rf))
        
        j = (idx+half)%N_FRAMES
        T_lf = np.eye(4)
        T_lf[:3,:3] = lf_rot
        T_lf[:3,3]  = [lf_st[0]+lf_dx[j], lf_st[1], lf_st[2]+lf_dz[j]]
        lf_t.set_target(mink.SE3.from_matrix(T_lf))

        T_pel_curr = T_pel.copy()
        T_pel_curr[2, 3] = target_z
        T_pel_curr[1, 3] += hip_sway * np.sin(2.0 * np.pi * t[idx])
        pel_t.set_target(mink.SE3.from_matrix(T_pel_curr))

        for _ in range(MAX_IT):
            vel = mink.solve_ik(cfg, tasks, DT, "daqp", limits=limits)
            cfg.integrate_inplace(vel, DT)
            mujoco.mj_kinematics(model, cfg.data)
            er = np.linalg.norm(cfg.data.site_xpos[rf_site_id]-T_rf[:3,3])
            el = np.linalg.norm(cfg.data.site_xpos[lf_site_id]-T_lf[:3,3])
            if er<TOL and el<TOL: break
        results.append(cfg.q.copy())
    return results

def save_csv(traj, save_path):
    cols = ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz"] + list(JMAP.keys())
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for qpos in traj:
            row = list(qpos[0:7])
            for jn in JMAP.keys():
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                row.append(qpos[model.jnt_qposadr[jid]])
            w.writerow(row)
    print(f"[SAVED] {len(traj)} frames → {save_path}")

# Anchor Precomputation System
anchor_speeds = np.linspace(v_base, v_transition, 10)
anchors_traj = []

def precompute_anchors():
    global waypoints
    print("⏳ Precomputing gait trajectories for anchor speeds...")
    frequencies = []
    for v in anchor_speeds:
        wps = get_waypoints_for_speed(v, rf_stance_world)
        waypoints = wps
        
        S = duty_factor(v)
        # Solve exact relationship: swing_speed = S / (1 - S) * stance_speed
        stance_speed = 1.0
        swing_speed = S / (1.0 - S)
        speeds = [swing_speed] * len(wps)
        
        L = stride_length(v)
        frequencies.append(v / L)
        
        print(f"  Speed {v:.3f} m/s | Stride Length: {L:.3f} m | Duty Factor: {S:.3f}")
        traj = solve_ik_traj(speeds, stance_speed, phase_frac=0.5, hip_sway=0.0)  # No sway: prevents legs from crossing over during swing
        anchors_traj.append(traj)
    print("✅ Precomputation complete!")
    
    # Extract the 12 joints in CSV order with flips for the NPZ
    npz_traj = []
    npz_base_pos = []
    npz_base_ori = []
    for traj in anchors_traj:
        processed_traj = []
        processed_base_pos = []
        processed_base_ori = []
        for qpos in traj:
            row = []
            for jn in JMAP.keys():
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                row.append(qpos[model.jnt_qposadr[jid]])
            processed_traj.append(row)
            processed_base_pos.append(qpos[0:3].copy())
            processed_base_ori.append(qpos[3:7].copy())
        npz_traj.append(processed_traj)
        npz_base_pos.append(processed_base_pos)
        npz_base_ori.append(processed_base_ori)
        
    npz_path = os.path.join(CURRENT_DIR, "angad_dynamic_gaits.npz")
    np.savez(
        npz_path,
        speeds=anchor_speeds,
        trajectories=np.array(npz_traj),
        base_positions=np.array(npz_base_pos),
        base_orientations=np.array(npz_base_ori),
        frequencies=np.array(frequencies),
        joint_names=list(JMAP.keys())
    )
    print(f"✅ Saved 12-DOF dynamic anchors, base pose trajectories, and frequencies to {npz_path}")

def interpolate_trajectory(v):
    # Clip v to the anchor range for interpolation
    v_clipped = np.clip(v, v_base, v_transition)
    
    idx = np.searchsorted(anchor_speeds, v_clipped)
    if idx == 0:
        return anchors_traj[0]
    if idx >= len(anchor_speeds):
        return anchors_traj[-1]
    
    v0 = anchor_speeds[idx - 1]
    v1 = anchor_speeds[idx]
    t = (v_clipped - v0) / (v1 - v0)
    
    traj0 = anchors_traj[idx - 1]
    traj1 = anchors_traj[idx]
    
    interpolated = []
    for f in range(N_FRAMES):
        q = (1.0 - t) * traj0[f] + t * traj1[f]
        interpolated.append(q)
    return interpolated


# Interactive GUI
class SpeedGaitMapperGUI(tk.Tk):
    def __init__(self, default_out):
        super().__init__()
        self.title("Angad Speed-to-Gait Mapper")
        self.geometry("450x600")
        self.configure(bg="#0f172a")
        
        self.speed_var = tk.DoubleVar(value=v_base)
        self.filename_var = tk.StringVar(value=default_out)
        self._viewer_running = False
        self._viewer_thread = None
        
        self._build_ui()
        self._update_readout()
        
    def _build_ui(self):
        # Title Label
        tk.Label(self, text="Angad Gait Synthesizer", bg="#0f172a", fg="#38bdf8",
                 font=("Segoe UI", 16, "bold")).pack(pady=15)
                 
        # Description
        desc = "Sweep target speed below to synthesize dynamically scaling walking gaits. Frequency and duty factor adapt using biomechanical models."
        tk.Label(self, text=desc, bg="#0f172a", fg="#94a3b8", wraplength=400,
                 justify="center", font=("Segoe UI", 9)).pack(pady=(0, 20))
        
        # Main Speed Slider
        tk.Label(self, text="Target Speed (v)", bg="#0f172a", fg="#f8fafc",
                 font=("Segoe UI", 10, "bold")).pack()
        self.slider = tk.Scale(self, variable=self.speed_var, from_=0.18, to=2.5, resolution=0.01,
                               orient="horizontal", length=350, bg="#0f172a", fg="#f8fafc",
                               troughcolor="#1e293b", highlightthickness=0, activebackground="#38bdf8",
                               command=lambda _: self._update_readout())
        self.slider.pack(pady=5)
        
        # Readout Frame
        rf = tk.LabelFrame(self, text=" Gait Metrics ", bg="#1e293b", fg="#38bdf8",
                            font=("Segoe UI", 10, "bold"), bd=1, relief="solid")
        rf.pack(fill="x", padx=25, pady=15)
        
        # Metric Labels
        self.lbl_v = tk.Label(rf, text="Speed: 0.18 m/s", bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 10))
        self.lbl_v.pack(anchor="w", padx=15, pady=4)
        
        self.lbl_L = tk.Label(rf, text="Stride Length: 0.320 m", bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 10))
        self.lbl_L.pack(anchor="w", padx=15, pady=4)
        
        self.lbl_F = tk.Label(rf, text="Stride Frequency: 0.566 Hz", bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 10))
        self.lbl_F.pack(anchor="w", padx=15, pady=4)
        
        self.lbl_S = tk.Label(rf, text="Duty Factor: 0.600", bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 10))
        self.lbl_S.pack(anchor="w", padx=15, pady=4)
        
        self.lbl_fr = tk.Label(rf, text="Froude Number: 0.004", bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 10))
        self.lbl_fr.pack(anchor="w", padx=15, pady=4)

        # Warning banner
        self.warn_banner = tk.Frame(self, bg="#ef4444", bd=0)
        self.warn_banner.pack(fill="x", padx=25, pady=(0, 10))
        self.warn_lbl = tk.Label(self.warn_banner, text="⚠️ EXTRAPOLATING (Froude >= 0.5)\nWalking geometry is invalid (requires flight phase).",
                                 bg="#ef4444", fg="white", font=("Segoe UI", 9, "bold"), justify="center")
        self.warn_lbl.pack(pady=4)
        self.warn_banner.pack_forget() # Hide by default

        # Output Name Field
        fn_frame = tk.Frame(self, bg="#0f172a")
        fn_frame.pack(fill="x", padx=25, pady=10)
        tk.Label(fn_frame, text="Save Filename:", bg="#0f172a", fg="#38bdf8",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Entry(fn_frame, textvariable=self.filename_var,
                 bg="#1e293b", fg="#f8fafc", borderwidth=0, insertbackground="white",
                 font=("Segoe UI", 10)).pack(fill="x", pady=2)
                 
        # Action Buttons
        btn_fr = tk.Frame(self, bg="#0f172a")
        btn_fr.pack(pady=20)
        
        tk.Button(btn_fr, text="▶ PLAY", command=self._play,
                  bg="#10b981", fg="white", font=("Segoe UI", 11, "bold"),
                  pady=8, padx=20, borderwidth=0, cursor="hand2").pack(side="left", padx=5)
                  
        tk.Button(btn_fr, text="■ STOP", command=self._stop,
                  bg="#ef4444", fg="white", font=("Segoe UI", 11, "bold"),
                  pady=8, padx=20, borderwidth=0, cursor="hand2").pack(side="left", padx=5)
                  
        tk.Button(btn_fr, text="💾 SAVE CSV", command=self._save,
                  bg="#3b82f6", fg="white", font=("Segoe UI", 11, "bold"),
                  pady=8, padx=20, borderwidth=0, cursor="hand2").pack(side="left", padx=5)

    def _update_readout(self):
        v = self.speed_var.get()
        L = stride_length(v)
        F = stride_frequency(v)
        S = duty_factor(v)
        
        # Froude number = v^2 / (g * leg_length)
        g = 9.81
        Fr = (v ** 2) / (g * leg_length)
        
        self.lbl_v.config(text=f"Speed: {v:.2f} m/s")
        self.lbl_L.config(text=f"Stride Length: {L:.3f} m")
        self.lbl_F.config(text=f"Stride Frequency: {F:.3f} Hz")
        self.lbl_S.config(text=f"Duty Factor (Stance %): {S:.3f}")
        self.lbl_fr.config(text=f"Froude Number (Fr): {Fr:.3f}")
        
        # Froude limits warning check
        if v > v_transition:
            self.warn_banner.pack(fill="x", padx=25, pady=(0, 10))
            self.slider.config(activebackground="#ef4444")
        else:
            self.warn_banner.pack_forget()
            self.slider.config(activebackground="#38bdf8")

    def _play(self):
        if self._viewer_running:
            return
        self._viewer_running = True
        
        def _run():
            m2 = mujoco.MjModel.from_xml_path(XML_PATH)
            d2 = mujoco.MjData(m2)
            gait_phase = 0.0
            last_time = time.time()
            
            with mujoco.viewer.launch_passive(m2, d2) as v_viewer:
                # Move camera to view the walking motion profile side-on
                v_viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
                v_viewer.cam.distance = 2.2
                v_viewer.cam.elevation = -12
                v_viewer.cam.azimuth = 90
                
                while v_viewer.is_running() and self._viewer_running:
                    # Read the speed dynamically from slider
                    current_v = self.speed_var.get()
                    traj = interpolate_trajectory(current_v)
                    
                    # Compute elapsed delta-time
                    now = time.time()
                    dt = now - last_time
                    last_time = now
                    dt = min(dt, 0.05) # Prevent high skipping
                    
                    # Advance gait phase based on a constant frequency for steady visual observation
                    F = 0.3
                    gait_phase = (gait_phase + dt * F) % 1.0
                    
                    # Select frame
                    idx = int(gait_phase * N_FRAMES) % N_FRAMES
                    
                    d2.qpos[:] = traj[idx]
                    mujoco.mj_forward(m2, d2)
                    v_viewer.sync()
                    time.sleep(0.005)
            self._viewer_running = False
            
        self._viewer_thread = threading.Thread(target=_run, daemon=True)
        self._viewer_thread.start()

    def _stop(self):
        self._viewer_running = False
        if self._viewer_thread:
            self._viewer_thread.join(timeout=1.0)

    def _save(self):
        current_v = self.speed_var.get()
        traj = interpolate_trajectory(current_v)
        
        fname = self.filename_var.get().strip()
        if not fname:
            fname = "angad_speed_mapped_gait.csv"
        if not fname.endswith(".csv"):
            fname += ".csv"
            
        if os.path.isabs(fname):
            save_path = fname
        else:
            save_path = os.path.abspath(os.path.join(CURRENT_DIR, "csvs", fname))
            
        save_csv(traj, save_path)
        messagebox.showinfo("Saved", f"Gait trajectory for {current_v:.2f} m/s successfully saved to:\n{save_path}")

def main():
    global model, data, rf_site_id, lf_site_id, stance_qpos, rf_stance_world, lf_stance_world
    
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="angad_speed_mapped_gait.csv", help="Default CSV save filename")
    args, _ = parser.parse_known_args()
    
    # Load and compile model
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    
    mujoco.mj_resetDataKeyframe(model, data, KEYFRAME_INDEX)
    mujoco.mj_forward(model, data)
    stance_qpos = data.qpos.copy()
    
    # Locate sites & stance coordinates
    rf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    if rf_site_id == -1:
        rf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_site")
    lf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    if lf_site_id == -1:
        lf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_site")
    rf_stance_world = data.site_xpos[rf_site_id].copy()
    lf_stance_world = data.site_xpos[lf_site_id].copy()
    
    # Run anchor precomputations on startup
    precompute_anchors()
    
    # Start GUI
    print("\n🏁 Starting Speed-to-Gait Mapper GUI...")
    app = SpeedGaitMapperGUI(args.out)
    app.mainloop()

if __name__ == "__main__":
    main()
