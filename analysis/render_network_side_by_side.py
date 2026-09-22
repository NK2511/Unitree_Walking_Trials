import os
import sys
import json
import math
from pathlib import Path
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def render_side_by_side():
    print("🎬 Rendering Side-by-Side Robot + Neural Network Video...")

    # Load network rollout data
    with open("neural_network_data.json", "r") as f:
        data = json.load(f)

    frames_data = data["frames"]
    total_frames = len(frames_data)

    # Open robot demo video
    cap = cv2.VideoCapture("unitree_g1_demo.mp4")
    if not cap.isOpened():
        raise FileNotFoundError("unitree_g1_demo.mp4 not found")

    output_path = "unitree_g1_network_telemetry.mp4"
    W, H = 1920, 1080
    fps = 50

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Neural network coordinates on the right half (from x = 960 to 1920)
    net_x_offset = 960
    net_w = 960
    net_h = H

    layer_x = [
        net_x_offset + 90,   # Layer 0 (Input 49)
        net_x_offset + 310,  # Layer 1 (Hidden 256)
        net_x_offset + 530,  # Layer 2 (Hidden 256)
        net_x_offset + 730,  # Layer 3 (Hidden 128)
        net_x_offset + 890   # Layer 4 (Output 12)
    ]

    top_y = 120
    bot_y = H - 80

    # Precompute node positions
    node_coords = []
    # Layer 0 (49 nodes)
    l0 = []
    for i in range(49):
        y = int(top_y + (i / 48) * (bot_y - top_y))
        l0.append((layer_x[0], y, 5))
    node_coords.append(l0)

    # Layer 1 (256 nodes in 4 columns of 64)
    l1 = []
    for i in range(256):
        col = i // 64
        row = i % 64
        x = int(layer_x[1] + (col - 1.5) * 22)
        y = int(top_y + (row / 63) * (bot_y - top_y))
        l1.append((x, y, 3))
    node_coords.append(l1)

    # Layer 2 (256 nodes in 4 columns of 64)
    l2 = []
    for i in range(256):
        col = i // 64
        row = i % 64
        x = int(layer_x[2] + (col - 1.5) * 22)
        y = int(top_y + (row / 63) * (bot_y - top_y))
        l2.append((x, y, 3))
    node_coords.append(l2)

    # Layer 3 (128 nodes in 2 columns of 64)
    l3 = []
    for i in range(128):
        col = i // 64
        row = i % 64
        x = int(layer_x[3] + (col - 0.5) * 26)
        y = int(top_y + (row / 63) * (bot_y - top_y))
        l3.append((x, y, 4))
    node_coords.append(l3)

    # Layer 4 (12 output nodes)
    l4 = []
    for i in range(12):
        y = int(top_y + 40 + (i / 11) * (bot_y - top_y - 80))
        l4.append((layer_x[4], y, 8))
    node_coords.append(l4)

    synapses = data["architecture"]["synapses"]

    for idx in range(total_frames):
        ret, robot_frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, robot_frame = cap.read()

        # Canvas frame
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (11, 15, 23) # Dark slate background

        # 1. Place robot frame on left side (scaled to 960x1080)
        # 1280x720 aspect ratio is 16:9. Resize to 960x540 and center vertically or resize to 960x1080 crop
        h_robot, w_robot = robot_frame.shape[:2]
        scaled_h = int(960 * (h_robot / w_robot)) # 540
        robot_resized = cv2.resize(robot_frame, (960, scaled_h))
        y_offset = (H - scaled_h) // 2
        canvas[y_offset:y_offset+scaled_h, 0:960] = robot_resized

        # Divider line
        cv2.line(canvas, (960, 0), (960, H), (40, 50, 70), 2)
        cv2.line(canvas, (960, 0), (960, H), (0, 242, 254), 1)

        # 2. Right Pane: Header
        cv2.putText(canvas, "NEURAL NETWORK ACTIVATION MAP", (net_x_offset + 40, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Brightness Proportional to ELU Neuron Excitation  |  PPO Policy", (net_x_offset + 40, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 1, cv2.LINE_AA)

        # Layer Labels
        layer_titles = ["Input (49)", "Hidden 1 (256)", "Hidden 2 (256)", "Hidden 3 (128)", "Output (12)"]
        for l_idx, lx in enumerate(layer_x):
            cv2.putText(canvas, layer_titles[l_idx], (lx - 45, top_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 210), 1, cv2.LINE_AA)

        frame_d = frames_data[idx]

        # 3. Draw Synapses
        def draw_syn(s_list, l1_idx, l2_idx):
            for in_j, out_i, w in s_list:
                p1 = node_coords[l1_idx][in_j]
                p2 = node_coords[l2_idx][out_i]
                act = frame_d[f"L{l1_idx}"][in_j]
                if act > 0.5:
                    alpha = min(1.0, act * 0.15)
                    color = (int(254 * alpha), int(242 * alpha), int(0 * alpha)) if w > 0 else (int(247 * alpha), int(85 * alpha), int(168 * alpha))
                    cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), color, 1, cv2.LINE_AA)

        draw_syn(synapses["s01"][:60], 0, 1)
        draw_syn(synapses["s12"][:60], 1, 2)
        draw_syn(synapses["s23"][:50], 2, 3)
        draw_syn(synapses["s34"][:30], 3, 4)

        # 4. Draw Neurons with Glow proportional to activation
        for l_idx in range(5):
            act_list = frame_d[f"L{l_idx}"]
            for n_idx, (nx, ny, nr) in enumerate(node_coords[l_idx]):
                val = act_list[n_idx]

                if l_idx == 4: # Output layer
                    if val > 0:
                        brightness = min(1.0, val * 1.5)
                        color = (int(50 + 150 * brightness), int(180 + 75 * brightness), 50)
                    else:
                        brightness = min(1.0, abs(val) * 1.5)
                        color = (50, 50, int(180 + 75 * brightness))
                else: # Hidden / Input
                    if val <= 0:
                        # Inhibited
                        color = (60, 45, 30) # dim slate
                    else:
                        # Excited: glowing neon cyan
                        t = min(1.0, val / 6.0)
                        b = int(254 * (0.3 + 0.7 * t))
                        g = int(242 * (0.3 + 0.7 * t))
                        r = int(50 * t)
                        color = (b, g, r) # BGR

                # Outer bloom if highly active
                if val > 2.0 or (l_idx == 4 and abs(val) > 0.4):
                    cv2.circle(canvas, (nx, ny), nr + 4, color, 1, cv2.LINE_AA)

                cv2.circle(canvas, (nx, ny), nr, color, -1, cv2.LINE_AA)

        # 5. Right-side Output Joint Labels
        out_names = [
            "L Hip Pitch", "L Hip Roll", "L Hip Yaw", "L Knee", "L Ankle P", "L Ankle R",
            "R Hip Pitch", "R Hip Roll", "R Hip Yaw", "R Knee", "R Ankle P", "R Ankle R"
        ]
        for o_i in range(12):
            ox, oy, _ = node_coords[4][o_i]
            val = frame_d["L4"][o_i]
            sign = "+" if val >= 0 else ""
            txt = f"{out_names[o_i]}: {sign}{val:.2f}"
            col = (100, 255, 120) if val >= 0 else (120, 100, 255)
            cv2.putText(canvas, txt, (ox - 140, oy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        # Bottom HUD on right
        t_sec = frame_d["t"]
        cmd_v = frame_d["cmd_vx"]
        act_v = frame_d["actual_vx"]
        dist = frame_d["distance"]
        phase = frame_d["gait_phase"]
        cv2.putText(canvas, f"Time: {t_sec:.2f}s | Speed Cmd: {cmd_v:.2f} m/s | Actual: {act_v:.2f} m/s | Distance: {dist:.2f} m | Gait Phase: {phase:.2f}",
                    (net_x_offset + 30, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 255), 1, cv2.LINE_AA)

        out.write(canvas)

        if idx % 50 == 0:
            sys.stdout.write(f"\r  Rendered frame {idx+1}/{total_frames} ({(idx+1)/total_frames*100:.1f}%)")
            sys.stdout.flush()

    cap.release()
    out.release()
    print(f"\n🎉 Saved side-by-side video to {output_path} ({Path(output_path).stat().st_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    render_side_by_side()
