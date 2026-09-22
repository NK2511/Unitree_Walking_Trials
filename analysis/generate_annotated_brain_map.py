import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def generate_annotated_brain_map():
    print("🧠 Building Anatomical Brain Weight Map with Labeled Sensory Input Clusters...")

    ckpt_path = Path("Unitree_Asimov_Trials/Walk/logs/rsl_rl/unitree_rigid_upper_body/2026-09-20_23-07-18/model_5499.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["actor_state_dict"]

    w0 = sd["mlp.0.weight"].numpy() # (256, 49)
    w2 = sd["mlp.2.weight"].numpy() # (256, 256)
    w4 = sd["mlp.4.weight"].numpy() # (128, 256)
    w6 = sd["mlp.6.weight"].numpy() # (12, 128)

    # Synaptic weight magnitudes (L1-norm)
    l0_power = np.abs(w0).sum(axis=0) # (49,)
    l1_power = np.abs(w0).sum(axis=1) + np.abs(w2).sum(axis=0) # (256,)
    l2_power = np.abs(w2).sum(axis=1) + np.abs(w4).sum(axis=0) # (256,)
    l3_power = np.abs(w4).sum(axis=1) + np.abs(w6).sum(axis=0) # (128,)
    l4_power = np.abs(w6).sum(axis=1) # (12,)

    # Glowing Bio-Colormap
    colors = [(0.04, 0.06, 0.12), (0.08, 0.35, 0.75), (0.0, 0.85, 0.95), (0.95, 0.75, 0.1), (1.0, 1.0, 1.0)]
    bio_cmap = LinearSegmentedColormap.from_list("bio_glow", colors, N=256)

    # Canvas Setup: 28 x 15 inches for high clarity and legible text
    fig, ax = plt.subplots(figsize=(28, 15), dpi=200, facecolor="#06080d")
    ax.set_facecolor("#06080d")

    # Layer X positions
    # Give Layer 0 generous room (0.05 to 0.28) for grouped bounding boxes & labels
    layer_x = [0.31, 0.48, 0.65, 0.81, 0.94]
    top_y, bot_y = 0.88, 0.08

    # =========================================================================
    # LAYER 0: ANATOMICAL SENSORY INPUT CLUSTERS WITH BORDERS & LABELS
    # =========================================================================
    l0_coords = {} # map channel idx -> (x, y)

    # Box 1: VESTIBULAR SYSTEM (IMU Gyro 0-2 & Projected Gravity 3-5)
    # Box 2: TASK COMMANDS & GAIT CLOCK (Commands 6-8 & Clock 47-48)
    # Box 3: PROPRIOCEPTION (Joint Pos 9-21 & Joint Vel 22-34) -> Left Leg vs Right Leg
    # Box 4: MOTOR MEMORY / PREV ACTIONS (35-46) -> Left Leg vs Right Leg

    # --- 1. PROPRIOCEPTION (Main center-left block) ---
    # We place Proprioception prominently in the middle (y: 0.18 to 0.64)
    prop_x_left = 0.085   # Left Leg Column
    prop_x_right = 0.175  # Right Leg Column
    prop_top = 0.62
    prop_bot = 0.19
    leg_joints = ["Hip Pitch", "Hip Roll", "Hip Yaw", "Knee", "Ankle Pitch", "Ankle Roll"]

    # Positions: Left leg (9-14), Right leg (15-20)
    # Velocities: Left leg (22-27), Right leg (28-33)
    # Total 12 points per column (6 Pos, 6 Vel)
    for i, jname in enumerate(leg_joints):
        y_p = prop_top - (i / 13) * (prop_top - prop_bot)
        # Left Leg Pos (idx: 9 + i)
        l0_coords[9 + i] = (prop_x_left, y_p)
        ax.text(prop_x_left - 0.008, y_p, f"L {jname} Pos", color="#cbd5e1", fontsize=7.2, va="center", ha="right", family="monospace")

        # Right Leg Pos (idx: 15 + i)
        l0_coords[15 + i] = (prop_x_right, y_p)
        ax.text(prop_x_right + 0.008, y_p, f"R {jname} Pos", color="#cbd5e1", fontsize=7.2, va="center", ha="left", family="monospace")

    # Add a subtle separator gap, then Joint Velocities below
    for i, jname in enumerate(leg_joints):
        y_v = prop_top - ((i + 6.8) / 13) * (prop_top - prop_bot)
        # Left Leg Vel (idx: 22 + i)
        l0_coords[22 + i] = (prop_x_left, y_v)
        ax.text(prop_x_left - 0.008, y_v, f"L {jname} Vel", color="#94a3b8", fontsize=7.2, va="center", ha="right", family="monospace")

        # Right Leg Vel (idx: 28 + i)
        l0_coords[28 + i] = (prop_x_right, y_v)
        ax.text(prop_x_right + 0.008, y_v, f"R {jname} Vel", color="#94a3b8", fontsize=7.2, va="center", ha="left", family="monospace")

    # Waist Yaw Pos (21) and Vel (34) centered at bottom of proprioception
    y_w1 = prop_top - (13.1 / 13) * (prop_top - prop_bot)
    y_w2 = prop_top - (13.8 / 13) * (prop_top - prop_bot)
    l0_coords[21] = ((prop_x_left + prop_x_right)/2, y_w1)
    l0_coords[34] = ((prop_x_left + prop_x_right)/2, y_w2)
    ax.text((prop_x_left + prop_x_right)/2 + 0.012, y_w1, "Waist Yaw Pos", color="#cbd5e1", fontsize=7.2, va="center", ha="left", family="monospace")
    ax.text((prop_x_left + prop_x_right)/2 + 0.012, y_w2, "Waist Yaw Vel", color="#94a3b8", fontsize=7.2, va="center", ha="left", family="monospace")

    # Column sub-headers inside Proprioception
    ax.text(prop_x_left, prop_top + 0.02, "LEFT LEG", color="#00f2fe", fontsize=8.5, fontweight="bold", ha="center")
    ax.text(prop_x_right, prop_top + 0.02, "RIGHT LEG", color="#00f2fe", fontsize=8.5, fontweight="bold", ha="center")

    # Bounding Box around Proprioception
    prop_box = patches.FancyBboxPatch(
        (0.015, prop_bot - 0.042), 0.235, (prop_top - prop_bot) + 0.095,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        edgecolor="#00f2fe", facecolor="none", lw=1.8, linestyle="-", zorder=2
    )
    ax.add_patch(prop_box)
    ax.text(0.028, prop_top + 0.045, "PROPRIOCEPTION (Joint Encoders: 26 Channels)", color="#00f2fe", fontsize=10.5, fontweight="bold", family="sans-serif")
    ax.text(0.028, prop_bot - 0.035, "Relative Joint Angles & Angular Velocities (Rad & Rad/s)", color="#64748b", fontsize=7.8, family="monospace")

    # --- 2. VESTIBULAR SYSTEM (Top Left: y: 0.72 to 0.88) ---
    vest_y_start = 0.86
    vest_dy = 0.024
    # Gyro X, Y, Z (0-2)
    gyro_names = ["IMU Gyro Roll (wx)", "IMU Gyro Pitch (wy)", "IMU Gyro Yaw (wz)"]
    for i in range(3):
        y = vest_y_start - i * vest_dy
        l0_coords[i] = (0.13, y)
        ax.text(0.12, y, gyro_names[i], color="#cbd5e1", fontsize=7.5, va="center", ha="right", family="monospace")

    # Projected Gravity X, Y, Z (3-5)
    grav_names = ["Proj Gravity X", "Proj Gravity Y", "Proj Gravity Z (Upright)"]
    for i in range(3):
        y = vest_y_start - (i + 3.2) * vest_dy
        l0_coords[3 + i] = (0.13, y)
        ax.text(0.12, y, grav_names[i], color="#38bdf8", fontsize=7.5, va="center", ha="right", family="monospace")

    # Bounding Box around Vestibular
    vest_box = patches.FancyBboxPatch(
        (0.015, vest_y_start - 5.5 * vest_dy - 0.015), 0.235, 5.5 * vest_dy + 0.04,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        edgecolor="#38bdf8", facecolor="none", lw=1.6, linestyle="--", zorder=2
    )
    ax.add_patch(vest_box)
    ax.text(0.028, vest_y_start + 0.02, "VESTIBULAR SYSTEM (Pelvis IMU: 6 Channels)", color="#38bdf8", fontsize=10, fontweight="bold", family="sans-serif")

    # --- 3. MOTOR MEMORY & EFFERENCE COPY (Bottom Left: y: 0.02 to 0.12) ---
    # Last Actions 35-46 (Left leg 35-40, Right leg 41-46)
    mem_top = 0.105
    mem_bot = 0.025
    for i, jname in enumerate(leg_joints):
        y_m = mem_top - (i / 5) * (mem_top - mem_bot)
        l0_coords[35 + i] = (prop_x_left, y_m)
        l0_coords[41 + i] = (prop_x_right, y_m)
        ax.text(prop_x_left - 0.008, y_m, f"Prev L {jname}", color="#f43f5e", fontsize=6.8, va="center", ha="right", family="monospace")
        ax.text(prop_x_right + 0.008, y_m, f"Prev R {jname}", color="#f43f5e", fontsize=6.8, va="center", ha="left", family="monospace")

    mem_box = patches.FancyBboxPatch(
        (0.015, mem_bot - 0.015), 0.235, (mem_top - mem_bot) + 0.045,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        edgecolor="#f43f5e", facecolor="none", lw=1.6, linestyle="--", zorder=2
    )
    ax.add_patch(mem_box)
    ax.text(0.028, mem_top + 0.022, "MOTOR MEMORY (Previous Actions at-1: 12 Channels)", color="#f43f5e", fontsize=9.5, fontweight="bold", family="sans-serif")

    # --- 4. COMMANDS & GAIT CLOCK (Bridge Column: x=0.27) ---
    cmd_top = 0.78
    cmd_names = ["Cmd Vel Vx", "Cmd Vel Vy", "Cmd Yaw Wz"]
    for i in range(3):
        y = cmd_top - i * 0.03
        l0_coords[6 + i] = (0.27, y)
        ax.text(0.262, y, cmd_names[i], color="#a855f7", fontsize=7.5, va="center", ha="right", family="monospace")

    clock_names = ["Gait Phase Cos", "Gait Phase Sin"]
    for i in range(2):
        y = cmd_top - (i + 3.4) * 0.03
        l0_coords[47 + i] = (0.27, y)
        ax.text(0.262, y, clock_names[i], color="#f59e0b", fontsize=7.5, va="center", ha="right", family="monospace")

    cmd_box = patches.FancyBboxPatch(
        (0.258, cmd_top - 4.6 * 0.03 - 0.012), 0.035, 4.6 * 0.03 + 0.045,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        edgecolor="#a855f7", facecolor="none", lw=1.6, linestyle=":", zorder=2
    )
    ax.add_patch(cmd_box)
    ax.text(0.275, cmd_top + 0.025, "GOAL & CLOCK\n(5 Channels)", color="#a855f7", fontsize=8.5, fontweight="bold", ha="center")

    # Combine all Layer 0 positions in sorted index order (0 to 48)
    coords_l0 = [l0_coords[i] for i in range(49)]

    # =========================================================================
    # HIDDEN LAYERS & OUTPUT LAYER
    # =========================================================================
    coords = [coords_l0]

    # Layer 1: 256 nodes (4 cols of 64)
    l1_coords = []
    for i in range(256):
        c = i // 64
        r = i % 64
        x = layer_x[1] + (c - 1.5) * 0.018
        y = top_y - (r / 63) * (top_y - bot_y)
        l1_coords.append((x, y))
    coords.append(l1_coords)

    # Layer 2: 256 nodes (4 cols of 64)
    l2_coords = []
    for i in range(256):
        c = i // 64
        r = i % 64
        x = layer_x[2] + (c - 1.5) * 0.018
        y = top_y - (r / 63) * (top_y - bot_y)
        l2_coords.append((x, y))
    coords.append(l2_coords)

    # Layer 3: 128 nodes (2 cols of 64)
    l3_coords = []
    for i in range(128):
        c = i // 64
        r = i % 64
        x = layer_x[3] + (c - 0.5) * 0.022
        y = top_y - (r / 63) * (top_y - bot_y)
        l3_coords.append((x, y))
    coords.append(l3_coords)

    # Layer 4: 12 output nodes
    l4_coords = []
    for i in range(12):
        y = top_y - 0.04 - (i / 11) * (top_y - bot_y - 0.08)
        l4_coords.append((layer_x[4], y))
    coords.append(l4_coords)

    # =========================================================================
    # DRAW SYNAPSES (Brightest Weights Glow Brightest)
    # =========================================================================
    def draw_synapses(W, c1, c2, max_lines=280):
        flat_idx = np.argsort(np.abs(W).ravel())[-max_lines:]
        max_w = np.abs(W).max()
        for idx in flat_idx:
            out_i, in_j = np.unravel_index(idx, W.shape)
            p1 = c1[in_j]
            p2 = c2[out_i]
            val = abs(W[out_i, in_j]) / max_w
            alpha = min(0.40, 0.06 + val * 0.34)
            color = (0.0, 0.85, 0.95, alpha) if W[out_i, in_j] > 0 else (0.85, 0.3, 0.95, alpha)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=0.35 + val * 1.0, zorder=1)

    draw_synapses(w0, coords[0], coords[1], max_lines=220)
    draw_synapses(w2, coords[1], coords[2], max_lines=320)
    draw_synapses(w4, coords[2], coords[3], max_lines=260)
    draw_synapses(w6, coords[3], coords[4], max_lines=130)

    # =========================================================================
    # DRAW NODES (Brightness proportional to L1-norm of learned weights)
    # =========================================================================
    all_powers = [l0_power, l1_power, l2_power, l3_power, l4_power]

    for l_idx, (layer_pts, powers) in enumerate(zip(coords, all_powers)):
        min_p = powers.min()
        max_p = powers.max()
        norm_p = (powers - min_p) / (max_p - min_p + 1e-8)

        base_radius = [20, 9, 9, 14, 50][l_idx]

        for i, (pt, p_val, norm_val) in enumerate(zip(layer_pts, powers, norm_p)):
            rgba = bio_cmap(norm_val)

            # Outer glow halo for highly developed neurons
            if norm_val > 0.45:
                halo_size = base_radius * (1.6 + norm_val * 2.0)
                ax.scatter(pt[0], pt[1], s=halo_size * 3.5, color=rgba, alpha=0.14 * norm_val, edgecolors="none", zorder=3)
                ax.scatter(pt[0], pt[1], s=halo_size * 1.8, color=rgba, alpha=0.28 * norm_val, edgecolors="none", zorder=4)

            # Core neuron
            core_size = base_radius * (0.85 + norm_val * 1.4)
            ax.scatter(pt[0], pt[1], s=core_size, color=rgba, alpha=0.90 + norm_val * 0.10, edgecolors="#ffffff", linewidths=0.35, zorder=5)

    # =========================================================================
    # HEADERS, LABELS, COLORBAR & MOTOR TARGET LABELS
    # =========================================================================
    layer_names = [
        "HIDDEN LAYER 1 (256)\nEarly Feature Extractors",
        "HIDDEN LAYER 2 (256)\nDeep Associative Core",
        "HIDDEN LAYER 3 (128)\nPremotor Controllers",
        "OUTPUT TARGETS (12)\nJoint Position Residuals"
    ]
    for lx, name in zip(layer_x[1:], layer_names):
        ax.text(lx, 0.94, name, color="#e2e8f0", fontsize=11, fontweight="bold", ha="center", va="center", family="sans-serif")

    # Output Joint Target Labels
    joint_targets = [
        "Left Hip Pitch", "Left Hip Roll", "Left Hip Yaw", "Left Knee", "Left Ankle Pitch", "Left Ankle Roll",
        "Right Hip Pitch", "Right Hip Roll", "Right Hip Yaw", "Right Knee", "Right Ankle Pitch", "Right Ankle Roll"
    ]
    for i, (lbl, pt) in enumerate(zip(joint_targets, coords[4])):
        norm_w = (l4_power[i] - l4_power.min()) / (l4_power.max() - l4_power.min() + 1e-8)
        color = bio_cmap(norm_w)
        ax.text(pt[0] + 0.012, pt[1], f"{lbl} ({l4_power[i]:.1f})", color=color, fontsize=9.5, fontweight="bold", va="center", family="monospace")

    # Main Title & Subtitle
    ax.text(0.015, 0.985, "UNITREE G1 — NEURAL NETWORK ARCHITECTURE & SYNAPTIC DEVELOPMENT", color="#ffffff", fontsize=18, fontweight="bold", va="top")
    ax.text(0.015, 0.963, "Trained PPO Policy (5,500 Iterations)  |  Every Neuron's Brightness is Proportional to Learned Synaptic Weight (L1-Norm)", color="#00f2fe", fontsize=11, va="top", family="monospace")

    # Bottom Colorbar
    sm = plt.cm.ScalarMappable(cmap=bio_cmap, norm=plt.Normalize(vmin=0, vmax=100))
    cbar_ax = fig.add_axes([0.52, 0.025, 0.26, 0.014])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 50, 100])
    cbar.set_ticklabels(["Dormant / Low Synaptic Weight", "Moderate Synaptogenesis", "Highly Developed Hub (Max Synaptic Mass)"])
    cbar.ax.tick_params(labelsize=8.5, colors="#cbd5e1")
    cbar.outline.set_edgecolor("#334155")

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    out_file = Path("neural_network_brain_map.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"🎉 Successfully generated updated {out_file.name}!")

if __name__ == "__main__":
    generate_annotated_brain_map()
