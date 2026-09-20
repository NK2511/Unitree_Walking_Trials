"""
verify_footstep_geometry.py
============================
Standalone MuJoCo diagnostic — visually verify that footstep targets are:

  1. Placed RELATIVE to the robot's spawn position (not at a fixed world origin)
  2. Placed IN FRONT of the robot (not behind it regardless of spawn heading)
  3. Left/Right assignment is correct (blue=left, red=right)
  4. Stance width (24 cm total) looks correct

HOW TO USE:
  Run this script, look at the MuJoCo viewer window.
  Every 8 seconds the robot is respawned at a NEW random yaw.
  Verify:
    - BLUE spheres are always on the robot's LEFT
    - RED  spheres are always on the robot's RIGHT
    - Markers start IN FRONT of the robot and continue forward
    - GREEN dot trail shows the robot's facing direction

SPHERE COLOURS:
  GREEN   = robot's spawn position / facing direction
  GOLD    = spline control points
  BLUE    = left foot targets  (bright = t1 active, dim = upcoming)
  RED     = right foot targets (bright = t1 active, dim = upcoming)
"""

import math
import os
import sys
import time

VENV_PYTHON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../mjlab_env/bin/python")
)
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    os.execl(VENV_PYTHON, VENV_PYTHON, *sys.argv)

import numpy as np
import mujoco
import mujoco.viewer

ROBOT_XML = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../xmls/Angad/scene.xml")
)

# Must match footstep_manager defaults exactly
STEP_WIDTH     = 0.12
STEP_DISTANCE  = 0.28
NUM_STEPS      = 20
NUM_SPLINE_PTS = 4
MIN_SEG        = 3.0
MAX_SEG        = 5.0
MAX_ANGLE_DEG  = 30.0


def quat_to_yaw(qw, qx, qy, qz):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def generate_steps(spawn_xy, spawn_yaw, first_is_left):
    """Replicate FootstepManager._generate_sequences_for geometry."""
    d_stride = STEP_DISTANCE / 2.0
    max_dev  = math.radians(MAX_ANGLE_DEG)
    rng      = np.random.default_rng()

    pts = [np.array(spawn_xy, dtype=float)]
    heading = spawn_yaw
    for _ in range(NUM_SPLINE_PTS):
        seg_len = rng.uniform(MIN_SEG, MAX_SEG)
        heading += rng.uniform(-max_dev, max_dev)
        pts.append(pts[-1] + seg_len * np.array([math.cos(heading), math.sin(heading)]))
    pts = np.array(pts)

    seg_dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_knots   = np.concatenate([[0.0], np.cumsum(seg_dists)])
    L_total   = s_knots[-1]

    footsteps = []
    for i in range(NUM_STEPS):
        s_val = (i + 1) * d_stride
        if s_val > L_total:
            break
        k  = min(np.searchsorted(s_knots[1:], s_val, side='left'), len(pts) - 2)
        s0, s1 = s_knots[k], s_knots[k + 1]
        u  = np.clip((s_val - s0) / max(s1 - s0, 1e-5), 0.0, 1.0)
        cx = pts[k, 0] + u * (pts[k+1, 0] - pts[k, 0])
        cy = pts[k, 1] + u * (pts[k+1, 1] - pts[k, 1])

        dx = pts[k+1, 0] - pts[k, 0]
        dy = pts[k+1, 1] - pts[k, 1]
        seg_norm = max(math.hypot(dx, dy), 1e-5)
        step_heading = math.atan2(dy / seg_norm, dx / seg_norm)
        nx, ny = -math.sin(step_heading), math.cos(step_heading)

        is_left = (i % 2 == 0) if first_is_left else (i % 2 != 0)
        y_side  = 1.0 if is_left else -1.0
        footsteps.append((cx + y_side * STEP_WIDTH * nx,
                          cy + y_side * STEP_WIDTH * ny,
                          step_heading, is_left))

    return pts, footsteps


def add_sphere(scene, pos, rgba, radius=0.04):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    g.type      = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[:]   = [radius, radius, radius]
    g.pos[:]    = pos
    g.mat[:,:]  = np.eye(3)          # shape (3,3) — do NOT flatten
    g.rgba[:]   = rgba
    g.dataid    = -1
    g.objtype   = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid     = -1
    g.category  = mujoco.mjtCatBit.mjCAT_DECOR
    g.segid     = -1
    scene.ngeom += 1



def run():
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    state = {"pts": None, "footsteps": None, "spawn_pos": None, "spawn_yaw": None}
    last_regen = [0.0]

    def regenerate():
        yaw = np.random.uniform(-math.pi, math.pi)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[3] = math.cos(yaw / 2)
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = math.sin(yaw / 2)
        mujoco.mj_forward(model, data)

        sp  = data.qpos[:2].copy()
        actual_yaw = quat_to_yaw(data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6])
        first_is_left = (np.random.rand() > 0.5)
        pts, footsteps = generate_steps(sp, actual_yaw, first_is_left)
        state.update({"pts": pts, "footsteps": footsteps,
                      "spawn_pos": sp, "spawn_yaw": actual_yaw})
        last_regen[0] = time.time()
        print(f"  Yaw: {math.degrees(actual_yaw):+.1f}°  |  First step: {'LEFT' if first_is_left else 'RIGHT'}  |  Steps: {len(footsteps)}")

    print("\nFootstep Geometry Verifier — respawns every 8s at a new random yaw")
    print("BLUE = LEFT foot   |   RED = RIGHT foot   |   GREEN trail = robot facing dir\n")
    regenerate()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 7.0
        viewer.cam.elevation = -35
        viewer.cam.azimuth   = 90

        while viewer.is_running():
            if time.time() - last_regen[0] > 8.0:
                regenerate()

            with viewer.lock():
                scene = viewer.user_scn
                scene.ngeom = 0

                pts       = state["pts"]
                footsteps = state["footsteps"]
                sp        = state["spawn_pos"]
                yaw       = state["spawn_yaw"]

                if pts is None:
                    continue

                # Spline control points
                ctrl_colors = [
                    (0.2, 0.9, 0.3, 1.0),
                    (1.0, 0.85, 0.0, 0.9),
                    (0.85, 0.1, 0.95, 0.9),
                    (1.0, 0.4, 0.0, 0.9),
                    (0.1, 0.9, 0.9, 0.9),
                ]
                for idx, pt in enumerate(pts):
                    r = 0.08 if idx == 0 else 0.05
                    add_sphere(scene, [pt[0], pt[1], 0.02],
                               ctrl_colors[idx % len(ctrl_colors)], radius=r)

                # Footstep markers
                for i, (fx, fy, heading, is_left) in enumerate(footsteps):
                    if is_left:
                        base = (0.1, 0.35, 1.0)   # blue
                    else:
                        base = (1.0, 0.1, 0.1)     # red

                    if i == 0:
                        rgba, radius = (*base, 1.0), 0.05     # t1 active
                    elif i == 1:
                        rgba, radius = (*base, 0.8), 0.038    # t2 upcoming
                    else:
                        rgba, radius = (base[0]*0.45, base[1]*0.45, base[2]*0.45, 0.5), 0.025

                    add_sphere(scene, [fx, fy, 0.01], rgba, radius=radius)

                # Robot facing direction (green dot trail)
                for t in np.linspace(0.05, 0.6, 10):
                    ax = sp[0] + math.cos(yaw) * t
                    ay = sp[1] + math.sin(yaw) * t
                    add_sphere(scene, [ax, ay, 0.04],
                               (0.0, 1.0, 0.0, 0.9), radius=0.015)

            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    run()
