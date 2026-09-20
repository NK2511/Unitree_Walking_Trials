

import argparse
import math
import os
import sys
import time

# Auto-switch to mjlab_env virtual environment if needed
VENV_PYTHON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../mjlab_env/bin/python")
)
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    os.execl(VENV_PYTHON, VENV_PYTHON, *sys.argv)

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import torch
from footstep_path_generator import PathGenerator, generate_footsteps_along_path

# Directly import LipmDcmTrajectoryGenerator without executing mdp/__init__.py
import importlib.util
lipm_spec = importlib.util.spec_from_file_location(
    "lipm_dcm_generator",
    os.path.join(os.path.dirname(__file__), "mdp/lipm_dcm_generator.py")
)
lipm_module = importlib.util.module_from_spec(lipm_spec)
lipm_spec.loader.exec_module(lipm_module)
LipmDcmTrajectoryGenerator = lipm_module.LipmDcmTrajectoryGenerator


DEFAULT_STEP_LENGTH     = 0.28   # Sagittal step distance along path centerline (meters)
DEFAULT_STEP_WIDTH      = 0.22   # Stance width offset perpendicular to path (meters)
DEFAULT_PATH_TYPE       = "rand_spline"  # Options: "straight", "sine", "circle", "rand_spline"
DEFAULT_NUM_STEPS       = 60      # Total number of footsteps to plan and execute
SINE_PATH_AMPLITUDE     = 0.40   # Peak lateral wave amplitude for sine path (meters)
SINE_PATH_WAVELENGTH    = 4.00   # Spatial wavelength for sine path (meters)
CIRCLE_PATH_RADIUS      = 2.50   # Path curvature radius for circle path (meters)
CIRCLE_PATH_ANGLE_DEG   = 270.0  # Total angular span for circle path (degrees)


RAND_SPLINE_NUM_POINTS         = 6      # Number of random control waypoints to chain
RAND_SPLINE_SEGMENT_DIST_RANGE = (1.0, 3.0)  # Segment distance range between waypoints (meters)
RAND_SPLINE_MAX_HEADING_DEG    = 30.0   # Max heading perturbation angle per waypoint (± degrees)
RAND_SPLINE_SEED               = None   # Optional seed for random path reproducibility

DEFAULT_SWING_ARC_HEIGHT = 0.12  # Peak vertical foot clearance height during swing (meters)
PELVIS_HEIGHT           = 0.831  # Target pelvis Z height in world frame (Stable_Stance keyframe, meters)
FOOT_SITE_Z             = 0.0644 # Foot site Z height above ground plane at touchdown (meters)
SWING_FRAMES            = 30    # Number of animation frames per swing phase (~0.5s to 1.0s)
DOUBLE_STANCE_FRAMES    = 2     # Pause duration (frames) in double-stance between steps

IK_ITERS                = 20     # Damped Least-Squares (DLS) IK iterations per frame
IK_DT                   = 0.01   # Integration step for IK updates (seconds)
IK_ALPHA_SWING          = 0.25   # Step size factor for swing leg joint updates
IK_ALPHA_STANCE         = 0.40   # Step size factor for stance leg joint updates
IK_DAMPING_SWING        = 0.05   # DLS regularization factor lambda for swing leg
IK_DAMPING_STANCE       = 0.02   # DLS regularization factor lambda for stance leg
IK_POS_WEIGHT_SWING     = 1.00   # Weight for swing foot position tracking
IK_ROT_WEIGHT_SWING     = 0.40   # Weight for swing foot orientation tracking (flat + heading)
IK_POS_WEIGHT_STANCE    = 1.00   # Weight for stance foot position tracking
IK_ROT_WEIGHT_STANCE    = 0.40   # Weight for stance foot orientation tracking (flat + landed heading)

RENDER_HZ               = 30     # Target playback framerate (frames per second)
SPHERE_RADIUS_ACTIVE    = 0.055  # Target marker sphere radius for active step (meters)
SPHERE_RADIUS_INACTIVE  = 0.040  # Target marker sphere radius for upcoming steps (meters)
ROBOT_XML_PATH          = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../xmls/Angad/scene.xml")
)

LEFT_JOINTS  = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw",
                "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll"]
RIGHT_JOINTS = ["right_hip_pitch", "right_hip_roll", "right_hip_yaw",
                 "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll"]


FLAT_XMAT = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
], dtype=np.float64)




def flat_xmat_for_heading(theta: float) -> np.ndarray:
    """Return flat foot orientation matrix rotated by heading angle theta (yaw) around world Z."""
    c, s_ = math.cos(theta), math.sin(theta)
    Rz = np.array([
        [c, -s_, 0.0],
        [s_,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return Rz @ FLAT_XMAT

def yaw_quat(yaw: float) -> np.ndarray:
    """Return quaternion [qw, qx, qy, qz] for a pure Z-axis yaw rotation."""
    return np.array([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)])

def swing_arc(p0: np.ndarray, p1: np.ndarray, s: float, h: float) -> np.ndarray:
    """Semicircular swing foot arc from p0 to p1 with peak clearance height h at s=0.5."""
    xy = (1.0 - s) * p0[:2] + s * p1[:2]
    z  = (1.0 - s) * p0[2]  + s * p1[2] + h * math.sin(math.pi * s)
    return np.array([xy[0], xy[1], z])

def ik_6dof_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    joint_ids: list,
    tgt_pos: np.ndarray,
    tgt_xmat: np.ndarray,
    pelvis_qpos: np.ndarray,
    pos_w: float = IK_POS_WEIGHT_SWING,
    rot_w: float = IK_ROT_WEIGHT_SWING,
    alpha: float = IK_ALPHA_SWING,
    damping: float = IK_DAMPING_SWING,
) -> float:
    """Combined 6-DOF position + orientation Damped Least-Squares IK for one leg with frozen pelvis."""
    data.qpos[:7] = pelvis_qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    cur_pos  = data.site_xpos[site_id].copy()
    cur_xmat = data.site_xmat[site_id].reshape(3, 3).copy()

    # Position error
    pos_err = tgt_pos - cur_pos

    # Rotation error (world-frame axis-angle)
    R_err    = cur_xmat.T @ tgt_xmat
    rot_b    = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ])
    rot_err  = cur_xmat @ rot_b

    # Jacobians
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, site_id)

    dof_ids = [model.jnt_dofadr[j] for j in joint_ids]
    J   = np.vstack([pos_w * Jp[:, dof_ids], rot_w * Jr[:, dof_ids]])
    err = np.concatenate([pos_w * pos_err, rot_w * rot_err])

    A  = J @ J.T + damping * np.eye(6)
    dq = J.T @ np.linalg.solve(A, err)

    for i, jid in enumerate(joint_ids):
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = np.clip(
            data.qpos[qadr] + alpha * dq[i],
            model.jnt_range[jid, 0],
            model.jnt_range[jid, 1],
        )
    return float(np.linalg.norm(pos_err))

def add_sphere(scn, pos: list, rgba: np.ndarray, r: float = 0.05):
    """Add a colored sphere marker to the MuJoCo viewer user scene."""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([r, 0.0, 0.0], dtype=np.float64),
        np.array(pos, dtype=np.float64),
        np.eye(3).flatten().astype(np.float64),
        rgba.astype(np.float32),
    )
    scn.ngeom += 1

def draw_spheres(
    scn,
    steps: list,
    active: int,
    current_zmp: np.ndarray = None,
    current_com: np.ndarray = None,
):
    """Draw all target spheres for the footstep plan; active step target is highlighted.
    Also draws dynamic ZMP (gold sphere) and LIPM CoM (white sphere) when provided.
    """
    scn.ngeom = 0
    for i, (foot, tx, ty, *_) in enumerate(steps):
        is_active = (i == active)
        if foot == "left":
            rgba = np.array([0.0, 0.7, 1.0, 1.0], np.float32) if is_active \
                   else np.array([0.2, 0.5, 1.0, 0.45], np.float32)
        else:
            rgba = np.array([1.0, 0.2, 0.2, 1.0], np.float32) if is_active \
                   else np.array([1.0, 0.3, 0.3, 0.45], np.float32)
        r = SPHERE_RADIUS_ACTIVE if is_active else SPHERE_RADIUS_INACTIVE
        add_sphere(scn, [tx, ty, 0.005], rgba, r)

    # Draw dynamic ZMP (Gold sphere on ground)
    if current_zmp is not None:
        add_sphere(
            scn,
            [current_zmp[0], current_zmp[1], 0.008],
            np.array([1.0, 0.84, 0.0, 0.9], np.float32),
            0.030,
        )

    # Draw dynamic LIPM CoM (White sphere at 3D CoM)
    if current_com is not None:
        add_sphere(
            scn,
            [current_com[0], current_com[1], current_com[2]],
            np.array([0.95, 0.95, 0.95, 0.85], np.float32),
            0.040,
        )

def execute_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    swing_site: int,
    other_site: int,
    swing_jids: list,
    other_jids: list,
    p_start: np.ndarray,
    p_end: np.ndarray,
    p_other: np.ndarray,
    yaw_start: float,
    yaw_end: float,
    swing_yaw_start: float,
    stance_yaw: float,
    frames: int,
    arc_h: float,
    all_steps: list,
    step_idx: int,
    frame_dt: float,
    lipm_gen: LipmDcmTrajectoryGenerator,
    t1_target: np.ndarray,
    t2_target: np.ndarray,
):
    """Execute a single swing step using PyTorch LIPM + DCM dynamically-balanced CoM and ZMP trajectories."""
    # Stance foot stays planted at its landed heading (no snapping)
    stance_xmat = flat_xmat_for_heading(stance_yaw)

    last_com = None
    last_zmp = None

    for frame in range(frames):
        if not viewer.is_running():
            return None, None, None

        s = frame / max(frames - 1, 1)

        # ---------------------------------------------------------------------
        # PyTorch Vectorized LIPM + DCM Analytical Trajectory Calculation
        # ---------------------------------------------------------------------
        phase_tensor = torch.tensor([s], dtype=torch.float32)
        p_stance_t = torch.from_numpy(p_other).float().unsqueeze(0)
        p_swing_start_t = torch.from_numpy(p_start).float().unsqueeze(0)
        t1_target_t = torch.from_numpy(t1_target).float().unsqueeze(0)
        t2_target_t = torch.from_numpy(t2_target).float().unsqueeze(0)

        p_com_t, v_com_t, zmp_ref_t = lipm_gen.compute_dynamic_com(
            phase_tensor, p_stance_t, p_swing_start_t, t1_target_t, t2_target_t
        )

        com_xyz = p_com_t[0].numpy()
        zmp_xy  = zmp_ref_t[0].numpy()
        last_com = com_xyz.copy()
        last_zmp = zmp_xy.copy()

        yaw = (1.0 - s) * yaw_start + s * yaw_end  # smooth pelvis heading interpolation

        # Dynamically-balanced 3D pelvis position from LIPM CoM
        pelvis_qpos = np.array([com_xyz[0], com_xyz[1], com_xyz[2], *yaw_quat(yaw)])

        tgt_swing = swing_arc(p_start, p_end, s, arc_h)

        # Swing foot orientation rotates smoothly in the air from swing_yaw_start -> yaw_end
        cur_swing_yaw = (1.0 - s) * swing_yaw_start + s * yaw_end
        swing_xmat = flat_xmat_for_heading(cur_swing_yaw)

        # Swing foot IK — position + orientation (flat, rotated smoothly to heading)
        for _ in range(IK_ITERS):
            ik_6dof_step(
                model, data, swing_site, swing_jids,
                tgt_swing, swing_xmat, pelvis_qpos,
                pos_w=IK_POS_WEIGHT_SWING, rot_w=IK_ROT_WEIGHT_SWING,
                alpha=IK_ALPHA_SWING, damping=IK_DAMPING_SWING,
            )

        # Stance foot IK — planted at p_other, flat at stance_yaw (no snapping)
        for _ in range(IK_ITERS):
            ik_6dof_step(
                model, data, other_site, other_jids,
                p_other, stance_xmat, pelvis_qpos,
                pos_w=IK_POS_WEIGHT_STANCE, rot_w=IK_ROT_WEIGHT_STANCE,
                alpha=IK_ALPHA_STANCE, damping=IK_DAMPING_STANCE,
            )

        # Freeze pelvis position & orientation
        data.qpos[:7] = pelvis_qpos
        mujoco.mj_kinematics(model, data)

        draw_spheres(viewer.user_scn, all_steps, step_idx, current_zmp=zmp_xy, current_com=com_xyz)
        viewer.sync()
        time.sleep(frame_dt)

    mujoco.mj_kinematics(model, data)
    final_pelvis_xy = last_com[:2] if last_com is not None else p_end[:2]
    return final_pelvis_xy, yaw_end, data.site_xpos[swing_site].copy()

def run(steps_spec: list, arc_h: float, frames: int):
    """Run kinematic walk for the sequence of steps in steps_spec using LIPM + DCM trajectories."""
    model = mujoco.MjModel.from_xml_path(ROBOT_XML_PATH)
    data  = mujoco.MjData(model)

    def jid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    def sid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,  n)

    left_jids  = [jid(n) for n in LEFT_JOINTS]
    right_jids = [jid(n) for n in RIGHT_JOINTS]
    l_site = sid("left_foot_site")
    r_site = sid("right_foot_site")

    # Load Stable_Stance keyframe (1)
    mujoco.mj_resetDataKeyframe(model, data, 1)
    for i in range(model.nmocap):
        data.mocap_pos[i] = [0.0, 0.0, -100.0]
    mujoco.mj_kinematics(model, data)

    # Track foot, pelvis, and per-foot landed yaws
    left_foot_pos   = data.site_xpos[l_site].copy()
    right_foot_pos  = data.site_xpos[r_site].copy()
    pelvis_xy       = np.array(data.qpos[:2])
    current_yaw     = 0.0   # Robot starts facing +X
    left_foot_yaw   = 0.0
    right_foot_yaw  = 0.0

    # Camera framing
    txs = [tx for _, tx, _, _ in steps_spec]
    tys = [ty for _, _, ty, _ in steps_spec]
    cam_x = sum(txs) / len(txs)
    cam_y = sum(tys) / len(tys)
    cam_d = max(abs(max(txs) - min(txs)), abs(max(tys) - min(tys)), 1.5) + 1.5

    frame_dt = 1.0 / RENDER_HZ
    step_duration = frames * frame_dt

    # Initialize PyTorch Vectorized LIPM + DCM Planner
    lipm_gen = LipmDcmTrajectoryGenerator(
        num_envs=1,
        device=torch.device("cpu"),
        com_height=PELVIS_HEIGHT,
        step_time=step_duration,
        dsp_ratio=0.20,
    )

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.lookat[:] = [cam_x * 0.5, cam_y * 0.5, 0.4]
    viewer.cam.distance  = cam_d
    viewer.cam.elevation = -20.0
    viewer.cam.azimuth   = 160.0

    try:
        for step_idx, (foot, tx, ty, theta) in enumerate(steps_spec):
            if not viewer.is_running():
                break

            p_end = np.array([tx, ty, FOOT_SITE_Z])

            # Prepare Target 1 (current step) & Target 2 (next step) for LIPM boundary conditions
            t1_target = np.array([tx, ty, FOOT_SITE_Z, theta], dtype=np.float32)
            if step_idx + 1 < len(steps_spec):
                next_s = steps_spec[step_idx + 1]
                t2_target = np.array([next_s[1], next_s[2], FOOT_SITE_Z, next_s[3]], dtype=np.float32)
            else:
                t2_target = t1_target.copy()

            if foot == "left":
                swing_site, other_site   = l_site, r_site
                swing_jids, other_jids   = left_jids, right_jids
                p_start = left_foot_pos.copy()
                p_other = right_foot_pos.copy()
                swing_yaw_start = left_foot_yaw
                stance_yaw      = right_foot_yaw
            else:
                swing_site, other_site   = r_site, l_site
                swing_jids, other_jids   = right_jids, left_jids
                p_start = right_foot_pos.copy()
                p_other = left_foot_pos.copy()
                swing_yaw_start = right_foot_yaw
                stance_yaw      = left_foot_yaw

            print(f"\n[Step {step_idx+1}] {foot.upper()} → ({tx:.3f}, {ty:.3f})  heading={math.degrees(theta):.1f}°")

            new_pelvis_xy, new_yaw, new_swing_pos = execute_step(
                model, data, viewer,
                swing_site, other_site,
                swing_jids, other_jids,
                p_start, p_end, p_other,
                current_yaw, theta,
                swing_yaw_start, stance_yaw,
                frames, arc_h,
                steps_spec, step_idx, frame_dt,
                lipm_gen, t1_target, t2_target,
            )

            if new_pelvis_xy is None:
                break

            pelvis_xy   = new_pelvis_xy
            current_yaw = new_yaw
            if foot == "left":
                left_foot_pos = new_swing_pos
                left_foot_yaw = theta
            else:
                right_foot_pos = new_swing_pos
                right_foot_yaw = theta

            # Brief double-stance pause
            for _ in range(DOUBLE_STANCE_FRAMES):
                if not viewer.is_running():
                    break
                draw_spheres(viewer.user_scn, steps_spec, step_idx)
                viewer.sync()
                time.sleep(frame_dt)

        print("\n[INFO] Walk complete — close viewer to exit.")
        while viewer.is_running():
            draw_spheres(viewer.user_scn, steps_spec, len(steps_spec) - 1)
            viewer.sync()
            time.sleep(frame_dt)

    finally:
        viewer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Kinematic walk demo for Angad.")
    p.add_argument("--type",      default=DEFAULT_PATH_TYPE, choices=["straight", "sine", "circle", "rand_spline"],
                   help="Path shape.")
    p.add_argument("--num-steps", type=int,   default=DEFAULT_NUM_STEPS,
                   help="Number of footsteps to walk.")
    p.add_argument("--step-len",  type=float, default=DEFAULT_STEP_LENGTH,
                   help="Step length along path centerline (m).")
    p.add_argument("--step-wid",  type=float, default=DEFAULT_STEP_WIDTH,
                   help="Stance width / lateral foot offset (m).")
    p.add_argument("--arc",       type=float, default=DEFAULT_SWING_ARC_HEIGHT,
                   help="Foot clearance arc peak height (m).")
    p.add_argument("--frames",    type=int,   default=SWING_FRAMES,
                   help="Animation frames per step.")
    a = p.parse_args()

    # Generate path waypoints
    path_len = a.num_steps * a.step_len + 0.5
    if a.type == "straight":
        waypoints = PathGenerator.straight_line(length=path_len)
    elif a.type == "sine":
        waypoints = PathGenerator.sine_wave(
            length=path_len, amplitude=SINE_PATH_AMPLITUDE, wavelength=SINE_PATH_WAVELENGTH
        )
    elif a.type == "circle":
        waypoints = PathGenerator.circle(
            radius=CIRCLE_PATH_RADIUS, angle_degrees=CIRCLE_PATH_ANGLE_DEG
        )
    elif a.type == "rand_spline":
        waypoints = PathGenerator.random_spline(
            num_control_points=RAND_SPLINE_NUM_POINTS,
            segment_dist_range=RAND_SPLINE_SEGMENT_DIST_RANGE,
            max_heading_offset_deg=RAND_SPLINE_MAX_HEADING_DEG,
            seed=RAND_SPLINE_SEED,
        )
    else:
        raise ValueError(f"Unknown path type: {a.type}")

    # Generate footsteps -> array [N, 5]: x, y, z, theta, side
    raw_steps = generate_footsteps_along_path(waypoints, a.step_len, a.step_wid)
    raw_steps = raw_steps[: a.num_steps]

    # Convert to (foot_str, tx, ty, theta)
    steps_spec = [
        ("left" if int(s[4]) == 0 else "right", float(s[0]), float(s[1]), float(s[3]))
        for s in raw_steps
    ]

    print(f"[INFO] Generated {len(steps_spec)} footsteps ({a.type} path)")
    for i, (foot, tx, ty, theta) in enumerate(steps_spec):
        print(f"  [{i+1}] {foot:5s}  ({tx:.3f}, {ty:.3f})  heading={math.degrees(theta):.1f}°")

    run(steps_spec, a.arc, a.frames)
