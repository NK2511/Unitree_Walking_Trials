#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import time

KEYFRAME_INDEX = 0

xml = "/home/nandhith/Python/Unitree_Walking_trials/xmls/unitree_g1/scene_mjx.xml"
model = mujoco.MjModel.from_xml_path(xml)
data  = mujoco.MjData(model)

print("\n--- Available Keyframes ---")
for i in range(model.nkey):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
    z    = model.key_qpos[i][2]
    print(f"  [{i}] {name}  (pelvis z = {z:.4f} m)")
print("---------------------------")
print("Controls: press 0-9 to switch keyframe | R to re-spawn\n")

current_kf     = [KEYFRAME_INDEX]
respawn        = [True]

def on_key(keycode):
    if 48 <= keycode <= 57:           # 0-9
        idx = keycode - 48
        if idx < model.nkey:
            current_kf[0] = idx
            respawn[0]    = True
    elif keycode in (82, 114):        # R / r
        respawn[0] = True

mujoco.mj_resetDataKeyframe(model, data, current_kf[0])

with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    while viewer.is_running():
        if respawn[0]:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, current_kf[0])
            print(f"📌 Spawning keyframe [{current_kf[0]}]: {name}")
            mujoco.mj_resetDataKeyframe(model, data, current_kf[0])
            data.qvel[:] = 0
            respawn[0] = False

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
