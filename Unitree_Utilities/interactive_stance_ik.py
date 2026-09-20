import os
import sys
# Auto-relaunch inside the correct virtual environment if not already using it
VENV_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Humanoid_Xterra_IITK/mjlab_env/bin/python"))
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    print(f"🔄 Auto-switching to mjlab_env Python interpreter...")
    os.execl(VENV_PYTHON, VENV_PYTHON, *sys.argv)

KEYFRAME_INDEX = 0

import numpy as np
import mujoco
import mujoco.viewer
import mink
import time

# Global state for keyboard interaction
model = None
data = None
all_mids = []
nmids = 4
curr_mid = 0

def keyboard_func(keycode):
    global all_mids, nmids, curr_mid, data, model

    incr_pos = 0.01  # metre
    incr_ang = 0.01  # radian

    # Toggle between mocap bodies using spacebar
    if keycode == 32:
        curr_mid = (curr_mid + 1) % nmids
        # Map mocap id back to body id to get the name
        body_id = model.body_mocapid.tolist().index(all_mids[curr_mid])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        print(f"Now controlling: {name}")
    elif keycode == 265:  # Up Arrow (Y decrease)
        data.mocap_pos[all_mids[curr_mid]][1] -= incr_pos
    elif keycode == 264:  # Down Arrow (Y increase)
        data.mocap_pos[all_mids[curr_mid]][1] += incr_pos
    elif keycode == 263:  # Left Arrow (X decrease)
        data.mocap_pos[all_mids[curr_mid]][0] -= incr_pos
    elif keycode == 262:  # Right Arrow (X increase)
        data.mocap_pos[all_mids[curr_mid]][0] += incr_pos
    elif keycode == 266:  # Page Up
        data.mocap_pos[all_mids[curr_mid]][2] += incr_pos
    elif keycode == 267:  # Page Down
        data.mocap_pos[all_mids[curr_mid]][2] -= incr_pos
    elif keycode == 49:  # 1
        dquat = np.array([np.cos(incr_ang / 2), np.sin(incr_ang / 2), 0., 0.])
        mujoco.mju_mulQuat(data.mocap_quat[all_mids[curr_mid]], data.mocap_quat[all_mids[curr_mid]], dquat)
    elif keycode == 50:  # 2
        dquat = np.array([np.cos(-incr_ang / 2), np.sin(-incr_ang / 2), 0., 0.])
        mujoco.mju_mulQuat(data.mocap_quat[all_mids[curr_mid]], data.mocap_quat[all_mids[curr_mid]], dquat)
    elif keycode == 51:  # 3
        dquat = np.array([np.cos(incr_ang / 2), 0., np.sin(incr_ang / 2), 0.])
        mujoco.mju_mulQuat(data.mocap_quat[all_mids[curr_mid]], data.mocap_quat[all_mids[curr_mid]], dquat)
    elif keycode == 52:  # 4
        dquat = np.array([np.cos(-incr_ang / 2), 0., np.sin(-incr_ang / 2), 0.])
        mujoco.mju_mulQuat(data.mocap_quat[all_mids[curr_mid]], data.mocap_quat[all_mids[curr_mid]], dquat)
    elif keycode == 53:  # 5
        dquat = np.array([np.cos(incr_ang / 2), 0., 0., np.sin(incr_ang / 2)])
        mujoco.mju_mulQuat(data.mocap_quat[all_mids[curr_mid]], data.mocap_quat[all_mids[curr_mid]], dquat)
    elif keycode == 54:  # 6
        dquat = np.array([np.cos(-incr_ang / 2), 0., 0., np.sin(-incr_ang / 2)])
        mujoco.mju_mulQuat(data.mocap_quat[all_mids[curr_mid]], data.mocap_quat[all_mids[curr_mid]], dquat)
    elif keycode == 75 or (32 <= keycode <= 126 and chr(keycode).lower() == 'k'):
        # Format the current qpos state to XML keyframe string for G1
        q = data.qpos
        base_pos = f"{q[0]:.6f}   {q[1]:.6f}   {q[2]:.6f}"
        base_quat = f"{q[3]:.6f}   {q[4]:.6f}   {q[5]:.6f}   {q[6]:.6f}"
        if len(q) == 20:
            l_leg = "  ".join(f"{val:.6f}" for val in q[7:13])
            r_leg = "  ".join(f"{val:.6f}" for val in q[13:19])
            waist = f"{q[19]:.6f}"
            xml_str = (
                f'    <key name="" qpos="\n'
                f'      {base_pos}\n'
                f'      {base_quat}\n'
                f'      {l_leg}\n'
                f'      {r_leg}\n'
                f'      {waist}\n'
                f'      " ctrl="\n'
                f'      {l_leg}\n'
                f'      {r_leg}\n'
                f'      {waist}\n'
                f'      "/>'
            )
        elif len(q) >= 36:
            l_leg = "  ".join(f"{val:.6f}" for val in q[7:13])
            r_leg = "  ".join(f"{val:.6f}" for val in q[13:19])
            waist = "  ".join(f"{val:.6f}" for val in q[19:22])
            l_arm = "  ".join(f"{val:.6f}" for val in q[22:29])
            r_arm = "  ".join(f"{val:.6f}" for val in q[29:36])
            xml_str = (
                f'    <key name="" qpos="\n'
                f'      {base_pos}\n'
                f'      {base_quat}\n'
                f'      {l_leg}\n'
                f'      {r_leg}\n'
                f'      {waist}\n'
                f'      {l_arm}\n'
                f'      {r_arm}\n'
                f'      "/>'
            )
        else:
            joints_str = "  ".join(f"{val:.6f}" for val in q[7:])
            xml_str = f'    <key name="" qpos="\n    {base_pos}\n    {base_quat}\n    {joints_str}\n    "/>'
        
        print("\n--- Copied Keyframe XML to Clipboard & Console ---")
        print(xml_str)
        print("--------------------------------------------------\n")
        
        try:
            import subprocess
            try:
                subprocess.run(["xclip", "-selection", "clipboard"], input=xml_str, text=True, check=True)
                print("Successfully copied to clipboard using xclip!")
            except Exception:
                subprocess.run(["wl-copy"], input=xml_str, text=True, check=True)
                print("Successfully copied to clipboard using wl-copy!")
        except Exception as e:
            print("Console copy ready (clipboard helper not found).")

def main():
    global model, data, all_mids

    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.abspath(os.path.join(current_dir, "../xmls/unitree_g1/scene_interactive_ik.xml"))
    
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        sys.exit(1)

    print(f"Loading model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Disable gravity and contacts for pure IK posing
    model.opt.gravity[:] = [0, 0, 0]
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    # Load keyframe 0 (home) if available
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, KEYFRAME_INDEX)
    mujoco.mj_forward(model, data)

    # Get mocap IDs
    mocap_names = ["base_target", "com_target", "left_foot_target", "right_foot_target", "zmp_target"]
    mocap_ids = []
    for name in mocap_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            print(f"Error: Mocap body '{name}' not found in model!")
            sys.exit(1)
        mocap_ids.append(model.body_mocapid[body_id])
    all_mids = mocap_ids

    # Setup Mink tasks
    posture_task = mink.PostureTask(model, cost=1.0)
    
    base_task = mink.FrameTask(
        frame_name="base_site",
        frame_type="site",
        position_cost=10.0,
        orientation_cost=50.0,
        lm_damping=1.0
    )

    torso_task = mink.FrameTask(
        frame_name="imu_in_torso",
        frame_type="site",
        position_cost=0.0,
        orientation_cost=100.0,
        lm_damping=1.0
    )
    
    com_task = mink.ComTask(cost=2.0)
    
    lfoot_task = mink.FrameTask(
        frame_name="left_foot",
        frame_type="site",
        position_cost=100.0,
        orientation_cost=50.0,
        lm_damping=1.0
    )
    
    rfoot_task = mink.FrameTask(
        frame_name="right_foot",
        frame_type="site",
        position_cost=100.0,
        orientation_cost=50.0,
        lm_damping=1.0
    )

    tasks_list = [base_task, torso_task, com_task, lfoot_task, rfoot_task, posture_task]
    configuration = mink.Configuration(model)
    configuration.update(data.qpos)

    # Snap mocaps to initial config
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    mink.move_mocap_to_frame(model, data, "base_target", "base_site", "site")
    mink.move_mocap_to_frame(model, data, "left_foot_target", "left_foot", "site")
    mink.move_mocap_to_frame(model, data, "right_foot_target", "right_foot", "site")
    
    # Com target doesn't map to a site, move directly to subtree COM
    data.mocap_pos[mocap_ids[1]] = data.subtree_com[0].copy()
    
    posture_task.set_target_from_configuration(configuration)

    print("\n--- INTERACTIVE IK SPAWN TUNING (Unitree G1) ---")
    print("Press SPACE to switch between targets (base, com, left foot, right foot).")
    print("Arrow keys: Move X/Y")
    print("PageUp/PageDown: Move Z")
    print("1/2: Pitch  |  3/4: Roll  |  5/6: Yaw")
    print("Press 'K' to copy the current joint state (qpos) to console/clipboard in XML keyframe format.")
    print(f"Initially controlling: {mocap_names[curr_mid]}\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_func,
                                      show_left_ui=True, show_right_ui=True) as viewer:
        
        # Configure group visibility:
        # Group 0: floor / base geoms
        # Group 1: target mocaps (COM, ZMP, foot/base targets)
        # Group 2: robot visual meshes (G1 visual meshes)
        # Group 3: collision primitives
        viewer.opt.geomgroup[0] = True
        viewer.opt.geomgroup[1] = True
        viewer.opt.geomgroup[2] = True
        viewer.opt.geomgroup[3] = False
        
        ik_dt = 1.0 / 60.0
        
        while viewer.is_running():
            step_start = time.time()
            
            # Sync IK targets to mocap body positions
            base_task.set_target(mink.SE3.from_mocap_id(data, mocap_ids[0]))
            torso_task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.identity(), np.zeros(3)))
            com_task.set_target(data.mocap_pos[mocap_ids[1]])
            lfoot_task.set_target(mink.SE3.from_mocap_id(data, mocap_ids[2]))
            rfoot_task.set_target(mink.SE3.from_mocap_id(data, mocap_ids[3]))

            # Solve IK
            vel = mink.solve_ik(configuration, tasks_list, ik_dt, "daqp", 1e-1)
            configuration.integrate_inplace(vel, ik_dt)
            data.qpos[:] = configuration.q
            
            mujoco.mj_forward(model, data)

            # COM Target Leashing
            actual_com = data.subtree_com[0].copy()
            if curr_mid != 1:  # 1 is com_target
                data.mocap_pos[mocap_ids[1]] = actual_com
            else:
                com_err = data.mocap_pos[mocap_ids[1]] - actual_com
                dist = np.linalg.norm(com_err)
                if dist > 0.05:  # max 5cm leash
                    data.mocap_pos[mocap_ids[1]] = actual_com + com_err / dist * 0.05

            # Set ZMP position (projection of actual COM onto floor)
            zmp_pos = actual_com.copy()
            zmp_pos[2] = 0.0
            data.mocap_pos[mocap_ids[4]] = zmp_pos

            # --- DRAW CONTACT POLYGON ---
            viewer.user_scn.ngeom = 0
            
            lf_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_box_collision")
            rf_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_box_collision")
            
            all_corners = []
            
            for geom_id in [lf_geom_id, rf_geom_id]:
                if geom_id >= 0:
                    geom_pos = data.geom_xpos[geom_id]
                    geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
                    geom_size = model.geom_size[geom_id]  # [dx, dy, dz]
                    dx, dy, dz = geom_size[0], geom_size[1], geom_size[2]
                    
                    local_bottom_corners = [
                        np.array([ dx,  dy, -dz]),
                        np.array([-dx,  dy, -dz]),
                        np.array([-dx, -dy, -dz]),
                        np.array([ dx, -dy, -dz])
                    ]
                    
                    for c in local_bottom_corners:
                        global_c = geom_pos + geom_mat.dot(c)
                        all_corners.append(global_c)

            # Compute 2D Convex Hull on floor (Z=0 plane)
            if len(all_corners) > 0:
                try:
                    from scipy.spatial import ConvexHull
                    points_2d = np.array([[c[0], c[1]] for c in all_corners])
                    hull = ConvexHull(points_2d)
                    hull_points = points_2d[hull.vertices]
                    
                    num_hull_pts = len(hull_points)
                    for i in range(num_hull_pts):
                        pt1 = np.array([hull_points[i][0], hull_points[i][1], 0.001])
                        pt2 = np.array([hull_points[(i + 1) % num_hull_pts][0], hull_points[(i + 1) % num_hull_pts][1], 0.001])
                        
                        if viewer.user_scn.ngeom < len(viewer.user_scn.geoms):
                            mujoco.mjv_connector(
                                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                mujoco.mjtGeom.mjGEOM_CAPSULE,
                                0.004,
                                pt1,
                                pt2
                            )
                            viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = [0, 1, 0, 0.8]
                            viewer.user_scn.ngeom += 1
                except Exception:
                    pass

            viewer.sync()
            
            elapsed = time.time() - step_start
            if elapsed < ik_dt:
                time.sleep(ik_dt - elapsed)

if __name__ == "__main__":
    main()
