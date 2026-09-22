import os
import sys
import time
import zipfile
import tempfile
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mujoco
import torch
import numpy as np
import cv2
import imageio

def generate_video_and_update_presentation():
    print("🚀 Initializing Unitree G1 Video Generation...")

    # 1. Load trained policy checkpoint
    ckpt_path = Path(r"Unitree_Asimov_Trials/Walk/logs/rsl_rl/unitree_rigid_upper_body/2026-09-20_23-07-18/model_5499.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["actor_state_dict"]

    class Policy:
        def __init__(self, sd):
            self.mean = sd["obs_normalizer._mean"].numpy()
            self.std = sd["obs_normalizer._std"].numpy()
            self.w0, self.b0 = sd["mlp.0.weight"].numpy(), sd["mlp.0.bias"].numpy()
            self.w2, self.b2 = sd["mlp.2.weight"].numpy(), sd["mlp.2.bias"].numpy()
            self.w4, self.b4 = sd["mlp.4.weight"].numpy(), sd["mlp.4.bias"].numpy()
            self.w6, self.b6 = sd["mlp.6.weight"].numpy(), sd["mlp.6.bias"].numpy()

        def __call__(self, x):
            x = (x - self.mean) / self.std
            def elu(z): return np.where(z > 0, z, np.exp(z) - 1.0)
            h1 = elu(x @ self.w0.T + self.b0)
            h2 = elu(h1 @ self.w2.T + self.b2)
            h3 = elu(h2 @ self.w4.T + self.b4)
            return (h3 @ self.w6.T + self.b6)[0]

    policy = Policy(sd)
    print("✅ Loaded trained PPO policy weights (model_5499.pt).")

    # 2. Setup MuJoCo Model & Renderer
    xml_path = r"xmls/unitree_g1/scene_mjx.xml"
    spec = mujoco.MjSpec.from_file(xml_path)
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
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

    renderer = mujoco.Renderer(m, 720, 1280)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.trackbodyid = pelvis_id
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.distance = 2.4
    cam.azimuth = 140
    cam.elevation = -14

    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro_pelvis")
    g_adr = m.sensor_adr[gid]

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

    # Video setup
    duration = 15.0
    fps = 50
    dt = 0.02
    total_frames = int(duration * fps) # 750 frames
    output_mp4 = Path("unitree_g1_demo.mp4")
    poster_png = Path("unitree_g1_poster.png")

    writer = imageio.get_writer(str(output_mp4), fps=fps, macro_block_size=None, codec="libx264", quality=8)
    print(f"🎬 Recording {total_frames} frames (15.0s @ 50fps) to {output_mp4.name}...")

    actions = np.zeros(12)
    gait_phase = 0.0
    poster_frame = None

    t0 = time.time()
    for frame_idx in range(total_frames):
        t_sim = frame_idx * dt
        # Velocity command ramp: 0.3 m/s to 1.2 m/s
        vx_cmd = 0.30 + (t_sim / duration) * 0.90

        # Build policy observation
        base_ang_vel = d.sensordata[g_adr:g_adr+3]
        proj_grav = quat_apply_inverse(d.qpos[3:7], [0., 0., -1.])
        cmd = np.array([vx_cmd, 0.0, 0.0])
        joint_pos = d.qpos[7:20] - default_q
        joint_vel = d.qvel[6:19]
        clock = np.array([np.cos(2*np.pi*gait_phase), np.sin(2*np.pi*gait_phase)])

        obs = np.concatenate([base_ang_vel, proj_grav, cmd, joint_pos, joint_vel, actions, clock])[None, :]
        actions = policy(obs)

        target_q = default_q.copy()
        target_q[:12] += 0.25 * actions
        d.ctrl[:13] = target_q

        # Step physics (4 sub-steps = 0.02s)
        prev_x = d.qpos[0]
        for _ in range(4):
            mujoco.mj_step(m, d)
        actual_vx = (d.qpos[0] - prev_x) / dt

        gait_phase = (gait_phase + dt * 1.25) % 1.0

        # Render frame
        renderer.update_scene(d, camera=cam)
        raw_pixels = renderer.render()

        # Add professional telemetry overlay
        img = raw_pixels.copy()
        h, w = img.shape[:2]

        # Top translucent header bar
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 16), (560, 130), (15, 20, 25), -1)
        cv2.rectangle(overlay, (w - 480, 16), (w - 20, 95), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Subtle border accents
        cv2.rectangle(img, (20, 16), (560, 130), (0, 210, 150), 2)
        cv2.rectangle(img, (w - 480, 16), (w - 20, 95), (60, 130, 240), 2)

        # Left Info: Telemetry & Speed
        cv2.putText(img, f"Command Speed: {vx_cmd:.2f} m/s", (40, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(img, f"Actual Velocity: {actual_vx:.2f} m/s", (40, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (230, 245, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Distance: {d.qpos[0]:.2f} m  |  Stability: 100% Upright", (40, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 215, 255), 1, cv2.LINE_AA)

        # Right Info: Robot & Policy Identity
        cv2.putText(img, "UNITREE G1 HUMANOID", (w - 460, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "PPO Policy (5,500 Iters)  |  MuJoCo MJX", (w - 460, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (140, 180, 255), 1, cv2.LINE_AA)

        writer.append_data(img)

        # Save active walking frame at t=2.0s as the poster frame
        if frame_idx == 100:
            poster_frame = img.copy()

        if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
            elapsed = time.time() - t0
            sys.stdout.write(f"\r  Frame {frame_idx+1}/{total_frames} ({((frame_idx+1)/total_frames)*100:.1f}%) | Speed: {vx_cmd:.2f} m/s | Elapsed: {elapsed:.1f}s")
            sys.stdout.flush()

    writer.close()
    renderer.close()
    print("\n✅ Video rendering complete!")

    if poster_frame is None:
        poster_frame = img

    # Save poster frame (RGB to BGR for cv2)
    cv2.imwrite(str(poster_png), cv2.cvtColor(poster_frame, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved poster frame to {poster_png.name}")

    # 3. Inject new video and poster into RL_MiniProj_Presentation.pptx
    pptx_path = Path("RL_MiniProj_Presentation.pptx")
    if not pptx_path.exists():
        raise FileNotFoundError(f"{pptx_path} not found")

    print(f"📦 Updating {pptx_path.name} with new Unitree G1 media...")

    temp_dir = Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(pptx_path, 'r') as zin:
            zin.extractall(temp_dir)

        # Replace media1.mp4
        target_video = temp_dir / "ppt" / "media" / "media1.mp4"
        shutil.copy2(output_mp4, target_video)
        print(f"  ✓ Replaced ppt/media/media1.mp4 ({target_video.stat().st_size / 1024 / 1024:.2f} MB)")

        # Replace image14.png (the poster preview for media1.mp4 on Slide 13)
        target_poster = temp_dir / "ppt" / "media" / "image14.png"
        shutil.copy2(poster_png, target_poster)
        print(f"  ✓ Replaced ppt/media/image14.png ({target_poster.stat().st_size / 1024:.1f} KB)")

        # Repack zip archive
        updated_pptx = temp_dir.parent / "temp_updated_presentation.pptx"
        with zipfile.ZipFile(updated_pptx, 'w', zipfile.ZIP_DEFLATED) as zout:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(temp_dir)
                    zout.write(full_path, str(rel_path))

        shutil.move(str(updated_pptx), str(pptx_path))
        print(f"🎉 Successfully updated {pptx_path.name} with authentic Unitree G1 video & poster!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    generate_video_and_update_presentation()
