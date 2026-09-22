#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import time

xml = "/home/nandhith/Python/Unitree_Walking_trials/xmls/unitree_g1/scene_mjx.xml"
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)  # "home" keyframe

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
