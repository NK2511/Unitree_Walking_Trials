#!/usr/bin/env python3
"""Angad Speed-to-Gait Cost-Optimized Mapping Tool.

This script implements speed-to-gait mapping by searching for the optimal
stride length (L) and frequency (F) that minimizes a physically-grounded
energy cost (Cost of Transport) at each speed.
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

import mujoco
import mujoco.viewer
import mink
from scipy.interpolate import PchipInterpolator
import tkinter as tk
from tkinter import messagebox

# ==============================================================================
# Configurable Hyperparameters & Constants for Unitree G1 (Top-level)
# ==============================================================================
LEG_LENGTH = 0.74        # m (Unitree G1 leg length from hip axis to sole)
MASS = 35.0              # kg (Unitree G1 total mass)
G = 9.81                 # m/s^2
V_TRANSITION = 1.906     # walk -> run transition speed (Froude = v^2 / (g*L) = 0.5)
SPEED_MIN = 0.1          # m/s (avoiding zero speed in optimization sweep)
SPEED_MAX = 2.5          # m/s
NUM_SPEED_SAMPLES = 12   # Number of speed points in the optimization sweep

# Optimization search grid configuration
NUM_L_CANDIDATES = 10    # Number of candidate stride lengths to test per speed
NUM_S_CANDIDATES = 5     # Number of candidate duty factors to test per speed

# Stride length search bounds (m)
L_MIN = 0.05
L_MAX = 0.80

# Stance duty factor bounds (Stance % / Cycle)
S_WALK_MIN = 0.5
S_WALK_MAX = 0.75
S_RUN_MIN = 0.4
S_RUN_MAX = 0.5

# GUI / NPZ anchor precomputation resolution
NUM_ANCHOR_SPEEDS = 15

# Simulation timestep
SIM_DT = 0.002           # s

# Joint PD control parameters for Unitree G1 motors (Hip/Knee/Ankle)
KP_LIST = np.array([100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 
                    100.0, 100.0, 100.0, 150.0, 40.0, 40.0])
KD_LIST = np.array([2.5, 2.5, 2.5, 4.0, 1.0, 1.0, 
                    2.5, 2.5, 2.5, 4.0, 1.0, 1.0])
TAU_SAT_LIST = np.array([88.0, 139.0, 88.0, 139.0, 50.0, 50.0, 
                         88.0, 139.0, 88.0, 139.0, 50.0, 50.0])

SWING_ASYMMETRY_OFFSET = 0.0  # Asymmetry offset for WP2 swing apex (default 0.0)
IK_TOLERANCE = 8e-4           # Max residual error tolerance for IK reachability
F_MAX = 3.5                   # Hz, maximum physically realistic step frequency
R_OVER_KT2 = 1.5              # Actuator torque cost coefficient (R / kt^2) for Joule heating loss

# File paths
XML_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../xmls/unitree_g1/scene.xml"))
if not os.path.exists(XML_PATH):
    XML_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../xmls/unitree_g1/g1.xml"))

OUT_CSV = os.path.abspath(os.path.join(CURRENT_DIR, "csvs/unitree_g1_speed_mapped_gait_cot.csv"))
KEYFRAME_INDEX = 0            # Index of keyframe to load for nominal standing stance
N_FRAMES = 120                # Number of frames in a gait trajectory cycle

# ==============================================================================
# Mapping Conventions & Global State
# ==============================================================================
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

model = data = None
rf_site_id = lf_site_id = None
stance_qpos = rf_stance_world = lf_stance_world = None

# Interpolator objects filled after optimization
L_interp = None
F_interp = None
CoT_interp = None
S_interp = None

# Precomputed anchors
anchor_speeds = None
anchors_traj = []

# ==============================================================================
# Biomechanical Models & Waypoint Generation
# ==============================================================================
def duty_factor(v):
    if S_interp is None:
        # Fallback linear shape if called before optimization completes
        t = np.clip((v - SPEED_MIN) / (SPEED_MAX - SPEED_MIN), 0.0, 1.0)
        return 0.6 + (0.4 - 0.6) * t
    return S_interp(v)

def cycloidal_warp(tau):
    return tau - np.sin(2 * np.pi * tau) / (2 * np.pi)

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

def get_waypoints_for_L(L):
    gnd = rf_stance_world[2]
    # WP1 (rear): x = -L/2
    wp1_pos = (rf_stance_world[0] - L / 2.0, rf_stance_world[1], gnd)
    # WP2 (swing apex): x is shifted forward by 9.4% of stride length to prevent stubbing, z is 5% of leg length
    wp2_x = 0.094 * L
    wp2_pos = (rf_stance_world[0] + wp2_x, rf_stance_world[1], gnd + 0.05 * LEG_LENGTH)
    # WP3 (front): x = +L/2
    wp3_pos = (rf_stance_world[0] + L / 2.0, rf_stance_world[1], gnd)
    
    return [
        {"pos": wp1_pos, "weight": 1.0},
        {"pos": wp2_pos, "weight": 1.0},
        {"pos": wp3_pos, "weight": 1.0}
    ]

# ==============================================================================
# IK Trajectory Generation & Reachability Checking
# ==============================================================================
def solve_ik_for_candidate(L, speeds, stance_speed, hip_sway=0.0):
    wps = get_waypoints_for_L(L)
    pts = np.array([w["pos"] for w in wps], dtype=float)
    gnd = rf_stance_world[2]
    pts[0, 2] = gnd
    pts[-1, 2] = gnd

    pc = np.vstack([pts, rf_stance_world, pts[0]])
    t_u = np.linspace(0, 1, len(pc))

    cs_x = PchipInterpolator(t_u, pc[:,0])
    cs_y = PchipInterpolator(t_u, pc[:,1])
    cs_z = PchipInterpolator(t_u, pc[:,2])
    
    t = temporal_t_map(speeds, stance_speed, N_FRAMES)
    tx, ty, tz = cs_x(t), cs_y(t), cs_z(t)
    
    stance_mask = t >= 0.5
    frac = 2.0 * (t[stance_mask] - 0.5)
    wp3_x = wps[2]["pos"][0]
    wp1_x = wps[0]["pos"][0]
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
    half = int(round(0.5 * N_FRAMES))

    cfg = mink.Configuration(model)
    cfg.update(stance_qpos)
    
    # Restored foot orientation constraint to keep feet flat
    rf_name = "right_foot" if mujoco.mj_name2id(m_tmp, mujoco.mjtObj.mjOBJ_SITE, "right_foot") != -1 else "right_foot_site"
    lf_name = "left_foot" if mujoco.mj_name2id(m_tmp, mujoco.mjtObj.mjOBJ_SITE, "left_foot") != -1 else "left_foot_site"
    rf_t = mink.FrameTask(rf_name, "site", position_cost=2000., orientation_cost=200., lm_damping=1e-2)
    lf_t = mink.FrameTask(lf_name, "site", position_cost=2000., orientation_cost=200., lm_damping=1e-2)
    
    # Restored base position constraint to prevent horizontal drift and wrap-around teleportation
    pel_t = mink.FrameTask("base", "body", position_cost=2000., orientation_cost=2000., lm_damping=1e-2)
    pel_t.set_target(mink.SE3.from_matrix(T_pel))
    
    post_t = mink.PostureTask(model, cost=1e-4)
    post_t.set_target(stance_qpos)
    tasks = [rf_t, lf_t, pel_t, post_t]
    limits = [mink.ConfigurationLimit(model)]
    DT, MAX_IT = 0.002, 100
    results = []
    
    # Calculate target pelvis height dynamically based on stride length to avoid leg over-extension
    max_reach = 0.78  # Safe maximum leg reach
    target_z = min(T_pel[2, 3], np.sqrt(max_reach**2 - (L / 2.0)**2))
    
    for idx in range(N_FRAMES):
        T_rf = np.eye(4)
        T_rf[:3,:3] = rf_rot
        T_rf[:3,3]  = [tx[idx], ty[idx], tz[idx]]
        rf_t.set_target(mink.SE3.from_matrix(T_rf))
        
        j = (idx + half) % N_FRAMES
        T_lf = np.eye(4)
        T_lf[:3,:3] = lf_rot
        T_lf[:3,3]  = [lf_st[0] + lf_dx[j], lf_st[1], lf_st[2] + lf_dz[j]]
        lf_t.set_target(mink.SE3.from_matrix(T_lf))

        T_pel_curr = T_pel.copy()
        T_pel_curr[2, 3] = target_z
        T_pel_curr[1, 3] += hip_sway * np.sin(2.0 * np.pi * t[idx])
        pel_t.set_target(mink.SE3.from_matrix(T_pel_curr))

        for _ in range(MAX_IT):
            vel = mink.solve_ik(cfg, tasks, DT, "daqp", limits=limits)
            cfg.integrate_inplace(vel, DT)
            mujoco.mj_kinematics(model, cfg.data)
            er = np.linalg.norm(cfg.data.site_xpos[rf_site_id] - T_rf[:3, 3])
            el = np.linalg.norm(cfg.data.site_xpos[lf_site_id] - T_lf[:3, 3])
            if er < IK_TOLERANCE and el < IK_TOLERANCE:
                break
        
        results.append(cfg.q.copy())
        
    return results, True

# ==============================================================================
# Cost Function Simulation
# ==============================================================================
def simulate_swing_cost(traj, F, S):
    T = 1.0 / F
    n_steps = max(200, int(T / SIM_DT))
    dt = T / n_steps
    
    # Save current simulator settings
    orig_gravity = model.opt.gravity.copy()
    orig_disable = model.opt.disableflags
    
    # Configure simulation for locked base swing energy evaluation
    mujoco.mj_resetData(model, data)
    model.opt.gravity[:] = [0.0, 0.0, -G]
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    
    data.qpos[:] = traj[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    
    power_t_array = []
    
    # Generate temporal map to get the warped phase mapping
    # This matches the phase logic used in solve_ik_for_candidate
    stance_speed = 1.0
    swing_speed = stance_speed * (3.0 - 1.0) / 2.0 * S / (1.0 - S)
    speeds = [swing_speed] * 3
    t_map = temporal_t_map(speeds, stance_speed, len(traj))
    
    for step in range(n_steps):
        t_curr = step * dt
        phase = (t_curr / T) % 1.0
        
        # Linearly interpolate configuration
        frame_idx = phase * len(traj)
        idx_low = int(np.floor(frame_idx)) % len(traj)
        idx_high = (idx_low + 1) % len(traj)
        alpha = frame_idx - np.floor(frame_idx)
        q_des = (1.0 - alpha) * traj[idx_low] + alpha * traj[idx_high]
        
        # Overwrite/lock base degrees of freedom in space (weld behavior)
        data.qpos[0:3] = stance_qpos[0:3]
        data.qpos[3:7] = stance_qpos[3:7]
        data.qvel[0:6] = 0.0
        
        # Compute PD commands
        for i in range(model.nu):
            jid = model.actuator_trnid[i][0]
            qpos_idx = model.jnt_qposadr[jid]
            qvel_idx = model.jnt_dofadr[jid]
            
            q = data.qpos[qpos_idx]
            qd = data.qvel[qvel_idx]
            
            tau = KP_LIST[i] * (q_des[qpos_idx] - q) + KD_LIST[i] * (0.0 - qd)
            tau = np.clip(tau, -TAU_SAT_LIST[i], TAU_SAT_LIST[i])
            data.ctrl[i] = tau
            
        mujoco.mj_step(model, data)
        
        # Calculate stance states from warped phase
        t_val = t_map[idx_low]
        right_stance = t_val >= 0.5
        left_stance = ((t_val + 0.5) % 1.0) >= 0.5
        
        # Distribute trunk gravity weight (MASS * G)
        F_g = MASS * G
        if right_stance and left_stance:
            F_r, F_l = 0.5 * F_g, 0.5 * F_g
        elif right_stance:
            F_r, F_l = F_g, 0.0
        elif left_stance:
            F_r, F_l = 0.0, F_g
        else:
            F_r, F_l = 0.0, 0.0
            
        # Get foot Jacobians (3 x nv)
        jacp_r = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp_r, None, rf_site_id)
        jacp_l = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp_l, None, lf_site_id)
        
        # Project vertical force to actuated joint torques (columns 6 to 18)
        tau_posture_r = jacp_r[2, 6:] * F_r
        tau_posture_l = jacp_l[2, 6:] * F_l
        
        # Integrate positive mechanical power and Joule heating losses (including postural heating)
        power_t = 0.0
        for i in range(model.nu):
            jid = model.actuator_trnid[i][0]
            qvel_idx = model.jnt_dofadr[jid]
            
            # Combine active joint torque and posture supporting torque
            # Since the motors produce both simultaneously, the total current/torque is the sum:
            # tau_total = tau_active + tau_posture
            actuator_force = data.actuator_force[i]
            qvel = data.qvel[qvel_idx]
            
            p_joint = actuator_force * qvel
            if p_joint > 0.0:
                power_t += p_joint
                
            # Add Joule heating/copper loss based on the total torque (active + posture)
            tau_posture_total = tau_posture_r[qvel_idx - 6] + tau_posture_l[qvel_idx - 6]
            p_joule = R_OVER_KT2 * ((actuator_force + tau_posture_total) ** 2)
            power_t += p_joule
        
        power_t_array.append(power_t)
        
    E_swing = np.sum(power_t_array) * dt
    
    # Restore original settings
    model.opt.gravity[:] = orig_gravity
    model.opt.disableflags = orig_disable
    
    return E_swing

# ==============================================================================
# Optimization Loop
# ==============================================================================
def run_optimization_sweep():
    print("=" * 60)
    print(" 🔋 RUNNING ANGLE COST-MINIMIZATION SEARCH ")
    print("=" * 60)
    opt_speeds = np.linspace(SPEED_MIN, SPEED_MAX, NUM_SPEED_SAMPLES)
    opt_L_res = []
    opt_F_res = []
    opt_CoT_res = []
    opt_S_res = []
    
    # 2D search grid setup
    L_candidates = np.linspace(L_MIN, L_MAX, NUM_L_CANDIDATES)
    
    for v in opt_speeds:
        # Filter L candidates so frequency does not exceed F_MAX
        valid_L = L_candidates[v / L_candidates <= F_MAX]
        if len(valid_L) == 0:
            valid_L = np.array([v / F_MAX])
        
        # Enforce walk/run duty factor boundaries based on Froude transition speed
        if v <= V_TRANSITION:
            # Walking speeds: S must be in [S_WALK_MIN, S_WALK_MAX]
            duty_candidates = np.linspace(S_WALK_MIN, S_WALK_MAX, NUM_S_CANDIDATES)
        else:
            # Running speeds: S can drop to [S_RUN_MIN, S_RUN_MAX] (forces flight phase)
            duty_candidates = np.linspace(S_RUN_MIN, S_RUN_MAX, NUM_S_CANDIDATES)
            
        best_L = None
        best_F = None
        best_S = None
        best_CoT = float('inf')
        
        for L_cand in valid_L:
            F_cand = v / L_cand
            
            for S_cand in duty_candidates:
                # Calculate swing speed based on the candidate S
                stance_speed = 1.0
                swing_speed = stance_speed * (3.0 - 1.0) / 2.0 * S_cand / (1.0 - S_cand)
                speeds = [swing_speed] * 3
                
                # 1. Reachability Check
                traj, ok = solve_ik_for_candidate(L_cand, speeds, stance_speed)
                if not ok:
                    continue
                    
                # 2. Swing Cost
                E_swing = simulate_swing_cost(traj, F_cand, S_cand)
                
                # 3. Collision Cost
                theta = np.arcsin((L_cand / 2.0) / LEG_LENGTH)
                E_collision = 2.0 * (0.5 * MASS * (v ** 2) * (np.sin(2.0 * theta) ** 2))
                
                # 4. Cost of Transport
                E_total = E_swing + E_collision 
                CoT = E_total / (MASS * G * L_cand)
                
                if CoT < best_CoT:
                    best_CoT = CoT
                    best_L = L_cand
                    best_F = F_cand
                    best_S = S_cand
                    
        if best_L is None:
            raise RuntimeError(f"Optimization failed: No reachable gait candidates found at speed v = {v:.3f} m/s!")
            
        print(f"  v = {v:.3f} m/s | Optimized L*={best_L:.3f} m, F*={best_F:.3f} Hz, S*={best_S:.3f}, CoT={best_CoT:.3f}")
            
        opt_L_res.append(best_L)
        opt_F_res.append(best_F)
        opt_CoT_res.append(best_CoT)
        opt_S_res.append(best_S)
        
    print("=" * 60)
    print("✅ Optimization search complete!")
    print("=" * 60)
    return opt_speeds, opt_L_res, opt_F_res, opt_CoT_res, opt_S_res

# ==============================================================================
# Spline & Interpolation Interpolators
# ==============================================================================
def setup_interpolators(opt_speeds, opt_L, opt_F, opt_CoT, opt_S):
    global L_interp, F_interp, CoT_interp, S_interp
    
    # Prepend 0.0 speed values for standing stance
    speeds = np.array([0.0] + list(opt_speeds))
    Ls = np.array([opt_L[0]] + list(opt_L))
    Fs = np.array([0.0] + list(opt_F))
    CoTs = np.array([0.0] + list(opt_CoT))
    Ss = np.array([opt_S[0]] + list(opt_S))
    
    L_interp = PchipInterpolator(speeds, Ls)
    F_interp = PchipInterpolator(speeds, Fs)
    CoT_interp = PchipInterpolator(speeds, CoTs)
    S_interp = PchipInterpolator(speeds, Ss)

def solve_ik_traj(v, hip_sway=0.0):
    if v < 1e-4:
        # Standing stance trajectory
        return [stance_qpos.copy() for _ in range(N_FRAMES)]
        
    L = L_interp(v)
    wps = get_waypoints_for_L(L)
    pts = np.array([w["pos"] for w in wps], dtype=float)
    gnd = rf_stance_world[2]
    pts[0, 2] = gnd
    pts[-1, 2] = gnd

    pc = np.vstack([pts, rf_stance_world, pts[0]])
    t_u = np.linspace(0, 1, len(pc))

    cs_x = PchipInterpolator(t_u, pc[:,0])
    cs_y = PchipInterpolator(t_u, pc[:,1])
    cs_z = PchipInterpolator(t_u, pc[:,2])
    
    S = duty_factor(v)
    stance_speed = 1.0
    swing_speed = stance_speed * (3.0 - 1.0) / 2.0 * S / (1.0 - S)
    speeds = [swing_speed] * 3
    
    t = temporal_t_map(speeds, stance_speed, N_FRAMES)
    tx, ty, tz = cs_x(t), cs_y(t), cs_z(t)
    
    stance_mask = t >= 0.5
    frac = 2.0 * (t[stance_mask] - 0.5)
    wp3_x = wps[2]["pos"][0]
    wp1_x = wps[0]["pos"][0]
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
    half = int(round(0.5 * N_FRAMES))

    cfg = mink.Configuration(model)
    cfg.update(stance_qpos)
    
    # Restored foot orientation constraint to keep feet flat
    rf_name = "right_foot" if mujoco.mj_name2id(m_tmp, mujoco.mjtObj.mjOBJ_SITE, "right_foot") != -1 else "right_foot_site"
    lf_name = "left_foot" if mujoco.mj_name2id(m_tmp, mujoco.mjtObj.mjOBJ_SITE, "left_foot") != -1 else "left_foot_site"
    rf_t = mink.FrameTask(rf_name, "site", position_cost=2000., orientation_cost=200., lm_damping=1e-2)
    lf_t = mink.FrameTask(lf_name, "site", position_cost=2000., orientation_cost=200., lm_damping=1e-2)
    
    # Restored base position constraint to prevent horizontal drift and wrap-around teleportation
    pel_t = mink.FrameTask("base", "body", position_cost=2000., orientation_cost=2000., lm_damping=1e-2)
    pel_t.set_target(mink.SE3.from_matrix(T_pel))
    
    post_t = mink.PostureTask(model, cost=1e-4)
    post_t.set_target(stance_qpos)
    tasks = [rf_t, lf_t, pel_t, post_t]
    limits = [mink.ConfigurationLimit(model)]
    
    DT, MAX_IT = 0.002, 100
    results = []
    
    # Calculate target pelvis height dynamically based on stride length to avoid leg over-extension
    max_reach = 0.78  # Safe maximum leg reach
    target_z = min(T_pel[2, 3], np.sqrt(max_reach**2 - (L / 2.0)**2))
    
    for idx in range(N_FRAMES):
        T_rf = np.eye(4)
        T_rf[:3,:3] = rf_rot
        T_rf[:3,3]  = [tx[idx], ty[idx], tz[idx]]
        rf_t.set_target(mink.SE3.from_matrix(T_rf))
        
        j = (idx + half) % N_FRAMES
        T_lf = np.eye(4)
        T_lf[:3,:3] = lf_rot
        T_lf[:3,3]  = [lf_st[0] + lf_dx[j], lf_st[1], lf_st[2] + lf_dz[j]]
        lf_t.set_target(mink.SE3.from_matrix(T_lf))

        T_pel_curr = T_pel.copy()
        T_pel_curr[2, 3] = target_z
        T_pel_curr[1, 3] += hip_sway * np.sin(2.0 * np.pi * t[idx])
        pel_t.set_target(mink.SE3.from_matrix(T_pel_curr))

        for _ in range(MAX_IT):
            vel = mink.solve_ik(cfg, tasks, DT, "daqp", limits=limits)
            cfg.integrate_inplace(vel, DT)
            mujoco.mj_kinematics(model, cfg.data)
            er = np.linalg.norm(cfg.data.site_xpos[rf_site_id] - T_rf[:3, 3])
            el = np.linalg.norm(cfg.data.site_xpos[lf_site_id] - T_lf[:3, 3])
            if er < IK_TOLERANCE and el < IK_TOLERANCE:
                break
        results.append(cfg.q.copy())
    return results

def precompute_anchors():
    global anchors_traj
    anchors_traj = []
    print("⏳ Precomputing gait trajectories for anchor speeds using optimized L*...")
    for v in anchor_speeds:
        S = duty_factor(v)
        print(f"  Anchor speed: {v:.3f} m/s | Stride Length: {L_interp(v):.3f} m | Duty Factor: {S:.3f}")
        traj = solve_ik_traj(v, hip_sway=0.0)
        anchors_traj.append(traj)
    print("✅ Precomputation complete!")
    
    # Save the anchors to NPZ
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
        
    npz_path = os.path.join(CURRENT_DIR, "csvs/gait_cot_dynamic_no_sway.npz")
    np.savez(
        npz_path,
        speeds=anchor_speeds,
        trajectories=np.array(npz_traj),
        base_positions=np.array(npz_base_pos),
        base_orientations=np.array(npz_base_ori),
        stride_lengths=L_interp(anchor_speeds),
        frequencies=F_interp(anchor_speeds),
        cots=CoT_interp(anchor_speeds),
        duty_factors=duty_factor(anchor_speeds),
        joint_names=list(JMAP.keys())
    )
    print(f"✅ Saved 12-DOF dynamic anchors, base pose trajectories, and all optimization metrics to {npz_path}")

def interpolate_trajectory(v):
    # Clip v to anchor range
    v_clipped = np.clip(v, SPEED_MIN, SPEED_MAX)
    
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

# ==============================================================================
# Interactive GUI
# ==============================================================================
class SpeedGaitMapperGUI(tk.Tk):
    def __init__(self, default_out):
        super().__init__()
        self.title("Angad Cost-Optimized Speed-to-Gait Mapper")
        self.geometry("450x640")
        self.configure(bg="#0f172a")
        
        self.speed_var = tk.DoubleVar(value=SPEED_MIN)
        self.filename_var = tk.StringVar(value=default_out)
        self._viewer_running = False
        self._viewer_thread = None
        
        self._build_ui()
        self._update_readout()
        
    def _build_ui(self):
        # Title Label
        tk.Label(self, text="Angad Gait Optimizer", bg="#0f172a", fg="#38bdf8",
                 font=("Segoe UI", 16, "bold")).pack(pady=15)
                 
        # Description
        desc = "Sweep target speed to play back cost-optimized gaits. The mapping tool uses a physics simulation grid search to minimize energy Cost of Transport."
        tk.Label(self, text=desc, bg="#0f172a", fg="#94a3b8", wraplength=400,
                 justify="center", font=("Segoe UI", 9)).pack(pady=(0, 20))
        
        # Main Speed Slider
        tk.Label(self, text="Target Speed (v)", bg="#0f172a", fg="#f8fafc",
                 font=("Segoe UI", 10, "bold")).pack()
        self.slider = tk.Scale(self, variable=self.speed_var, from_=SPEED_MIN, to=SPEED_MAX, resolution=0.01,
                               orient="horizontal", length=350, bg="#0f172a", fg="#f8fafc",
                               troughcolor="#1e293b", highlightthickness=0, activebackground="#38bdf8",
                               command=lambda _: self._update_readout())
        self.slider.pack(pady=5)
        
        # Readout Frame
        rf = tk.LabelFrame(self, text=" Cost-Optimized Gait Metrics ", bg="#1e293b", fg="#38bdf8",
                            font=("Segoe UI", 10, "bold"), bd=1, relief="solid")
        rf.pack(fill="x", padx=25, pady=15)
        
        # Metric Labels
        self.lbl_v = tk.Label(rf, text="Speed: -- m/s", bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 10))
        self.lbl_v.pack(anchor="w", padx=15, pady=4)
        
        self.lbl_L = tk.Label(rf, text="Stride Length (L*): -- m", bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 10))
        self.lbl_L.pack(anchor="w", padx=15, pady=4)
        
        self.lbl_F = tk.Label(rf, text="Stride Frequency (F*): -- Hz", bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 10))
        self.lbl_F.pack(anchor="w", padx=15, pady=4)
        
        self.lbl_S = tk.Label(rf, text="Duty Factor (Stance %): --", bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 10))
        self.lbl_S.pack(anchor="w", padx=15, pady=4)
        
        self.lbl_cot = tk.Label(rf, text="Cost of Transport (CoT): --", bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 10))
        self.lbl_cot.pack(anchor="w", padx=15, pady=4)
        
        self.lbl_fr = tk.Label(rf, text="Froude Number (Fr): --", bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 10))
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
        L = L_interp(v)
        F = F_interp(v)
        S = duty_factor(v)
        CoT = CoT_interp(v)
        
        Fr = (v ** 2) / (G * LEG_LENGTH)
        
        self.lbl_v.config(text=f"Speed: {v:.2f} m/s")
        self.lbl_L.config(text=f"Stride Length (L*): {L:.3f} m")
        self.lbl_F.config(text=f"Stride Frequency (F*): {F:.3f} Hz")
        self.lbl_S.config(text=f"Duty Factor (Stance %): {S:.3f}")
        self.lbl_cot.config(text=f"Cost of Transport (CoT): {CoT:.3f}")
        self.lbl_fr.config(text=f"Froude Number (Fr): {Fr:.3f}")
        
        # Froude limits warning check
        if v > V_TRANSITION:
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
                v_viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
                v_viewer.cam.distance = 2.2
                v_viewer.cam.elevation = -12
                v_viewer.cam.azimuth = 90
                
                while v_viewer.is_running() and self._viewer_running:
                    current_v = self.speed_var.get()
                    traj = interpolate_trajectory(current_v)
                    
                    now = time.time()
                    dt = now - last_time
                    last_time = now
                    dt = min(dt, 0.05)
                    
                    F = 0.3
                    gait_phase = (gait_phase + dt * F) % 1.0
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
            fname = "angad_speed_mapped_gait_new.csv"
        if not fname.endswith(".csv"):
            fname += ".csv"
            
        if os.path.isabs(fname):
            save_path = fname
        else:
            save_path = os.path.abspath(os.path.join(CURRENT_DIR, "csvs", fname))
            
        save_csv(traj, save_path)
        messagebox.showinfo("Saved", f"Gait trajectory for {current_v:.2f} m/s successfully saved to:\n{save_path}")

# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    global model, data, rf_site_id, lf_site_id, stance_qpos, rf_stance_world, lf_stance_world
    
    # Setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="angad_speed_mapped_gait_new.csv", help="Default CSV save filename")
    args, _ = parser.parse_known_args()
    
    # Initialize MuJoCo environment
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    
    mujoco.mj_resetDataKeyframe(model, data, KEYFRAME_INDEX)
    mujoco.mj_forward(model, data)
    stance_qpos = data.qpos.copy()
    
    # Find feet site locations
    rf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    if rf_site_id == -1:
        rf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_site")
    lf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    if lf_site_id == -1:
        lf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_site")
    rf_stance_world = data.site_xpos[rf_site_id].copy()
    lf_stance_world = data.site_xpos[lf_site_id].copy()
    
    # 1. Run cost minimization optimization sweep
    opt_speeds, opt_L, opt_F, opt_CoT, opt_S = run_optimization_sweep()
    
    # 2. Build continuous interpolators
    global anchor_speeds
    setup_interpolators(opt_speeds, opt_L, opt_F, opt_CoT, opt_S)
    
    # 3. Precompute anchors for GUI slider interpolation & export NPZ
    anchor_speeds = np.linspace(SPEED_MIN, SPEED_MAX, NUM_ANCHOR_SPEEDS)
    precompute_anchors()
    
    # 4. Start interactive GUI
    print("\n🏁 Starting Speed-to-Gait Optimizer GUI...")
    app = SpeedGaitMapperGUI(args.out)
    app.mainloop()

if __name__ == "__main__":
    main()
