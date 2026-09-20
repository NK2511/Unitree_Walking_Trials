import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import csv
import os

# Configurable default path when running without arguments
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GAIT_FILE = os.path.join(CURRENT_DIR, "csvs", "unitree_g1_speed_mapped_gait_cot.csv")
if not os.path.exists(DEFAULT_GAIT_FILE):
    DEFAULT_GAIT_FILE = os.path.join(CURRENT_DIR, "csvs", "unitree_g1_speed_mapped_gait.csv")

def get_lowest_geom_z(model, data, body_names):
    min_z = float('inf')
    for bname in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid == -1:
            continue
        geom_start = model.body_geomadr[bid]
        geom_num = model.body_geomnum[bid]
        for g_idx in range(geom_start, geom_start + geom_num):
            g_type = model.geom_type[g_idx]
            g_pos = data.geom_xpos[g_idx]
            g_mat = data.geom_xmat[g_idx].reshape(3, 3)
            g_size = model.geom_size[g_idx]
            
            if g_type == mujoco.mjtGeom.mjGEOM_BOX:
                offset = np.sum(np.abs(g_mat[2, :] * g_size[:3]))
                z_bottom = g_pos[2] - offset
            elif g_type in [mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE]:
                r, h = g_size[0], g_size[1]
                offset = r * np.sqrt(g_mat[2, 0]**2 + g_mat[2, 1]**2) + np.abs(g_mat[2, 2]) * h
                z_bottom = g_pos[2] - offset
            elif g_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                z_bottom = g_pos[2] - g_size[0]
            else:
                z_bottom = g_pos[2]
            
            min_z = min(min_z, z_bottom)
    return min_z

def play_gait(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found -> {filepath}")
        return

    xml_path = os.path.abspath(os.path.join(CURRENT_DIR, "../../xmls/unitree_g1/scene.xml"))
    if not os.path.exists(xml_path):
        xml_path = os.path.abspath(os.path.join(CURRENT_DIR, "../../xmls/unitree_g1/g1.xml"))
            
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print(f"Loading {filepath}...")
    
    trajectory = []
    base_positions = None
    base_orientations = None
    
    if filepath.endswith('.npz'):
        npz_data = np.load(filepath)
        trajectories = npz_data['trajectories']
        speeds = npz_data['speeds']
        # The NPZ arrays are always saved in JMAP order: [Left Leg (0-5), Right Leg (6-11)]
        csv_to_mjlab = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
        
        speed_idx = len(speeds) - 1
        print(f"Playing trajectory for fastest speed: {speeds[speed_idx]:.2f} m/s")
        
        raw_traj = trajectories[speed_idx]
        trajectory = raw_traj[:, csv_to_mjlab]
        
        if 'base_positions' in npz_data and 'base_orientations' in npz_data:
            base_positions = npz_data['base_positions'][speed_idx]
            base_orientations = npz_data['base_orientations'][speed_idx]
            print("[INFO] Found and loaded base pose trajectories from NPZ.")
        
    elif filepath.endswith('.csv'):
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
            qpos_indices = []
            for h in headers:
                j_name = h
                if j_name == "left_knee": j_name = "left_knee_pitch"
                if j_name == "right_knee": j_name = "right_knee_pitch"
                
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
                if jid != -1:
                    qpos_indices.append((headers.index(h), model.jnt_qposadr[jid] - 7)) 
                
            base_pos_indices = [headers.index(h) if h in headers else -1 for h in ["base_x", "base_y", "base_z"]]
            base_ori_indices = [headers.index(h) if h in headers else -1 for h in ["base_qw", "base_qx", "base_qy", "base_qz"]]
            has_base = all(idx != -1 for idx in base_pos_indices + base_ori_indices)
            
            if has_base:
                base_positions = []
                base_orientations = []
                print("[INFO] Found base pose headers in CSV. Loading base trajectory.")
                
            for row in reader:
                if not row: continue
                frame_data = np.zeros(12)
                for col_idx, q_idx in qpos_indices:
                    frame_data[q_idx] = float(row[col_idx])
                trajectory.append(frame_data)
                
                if has_base:
                    base_positions.append([float(row[idx]) for idx in base_pos_indices])
                    base_orientations.append([float(row[idx]) for idx in base_ori_indices])
                    
            if has_base:
                base_positions = np.array(base_positions)
                base_orientations = np.array(base_orientations)
    else:
        print("Unsupported format! Pass a .csv or .npz")
        return

    left_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_site")
    right_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_site")

    print("Launching GUI... (Close the window to exit)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -15
        viewer.cam.distance = 2.5
        viewer.cam.lookat[:] = [0, 0, 0.8]
        
        while viewer.is_running():
            for i in range(len(trajectory)):
                if not viewer.is_running():
                    break
                    
                if base_positions is not None and base_orientations is not None:
                    data.qpos[0:3] = base_positions[i]
                    data.qpos[3:7] = base_orientations[i]
                else:
                    data.qpos[0:3] = [0, 0, 0.95]
                    data.qpos[3:7] = [1, 0, 0, 0]
                    
                data.qpos[7:19] = trajectory[i]
                
                mujoco.mj_forward(model, data)
                
                # Dynamic grounding: shift the base Z so the lowest foot geometry is exactly on the floor (Z=0)
                lowest_foot_z = get_lowest_geom_z(model, data, ["left_foot_link", "right_foot_link"])
                if lowest_foot_z != float('inf'):
                    if base_positions is not None:
                        data.qpos[2] = base_positions[i][2] - lowest_foot_z
                    else:
                        data.qpos[2] = 0.95 - lowest_foot_z
                    mujoco.mj_forward(model, data)
                
                # Keep the camera tracking the robot base if base moves forward
                viewer.cam.lookat[:] = data.qpos[0:3] + np.array([0, 0, -0.15])
                
                viewer.sync()
                
                time.sleep(0.02) # Plays in slow motion (approx 4x slower than realtime sprint)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"No file path provided. Defaulting to: {DEFAULT_GAIT_FILE}")
        play_gait(DEFAULT_GAIT_FILE)
    else:
        play_gait(sys.argv[1])
