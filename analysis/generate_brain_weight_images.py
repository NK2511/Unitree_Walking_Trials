import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def generate_weight_visualizations():
    print("🧠 Analyzing Neural Network Synaptic Weights...")

    ckpt_path = Path("Unitree_Asimov_Trials/Walk/logs/rsl_rl/unitree_rigid_upper_body/2026-09-20_23-07-18/model_5499.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["actor_state_dict"]

    w0 = sd["mlp.0.weight"].numpy() # (256, 49)
    w2 = sd["mlp.2.weight"].numpy() # (256, 256)
    w4 = sd["mlp.4.weight"].numpy() # (128, 256)
    w6 = sd["mlp.6.weight"].numpy() # (12, 128)

    # Calculate neuron importance (total absolute synaptic strength per neuron)
    # Layer 0 (Input 49): outgoing weights
    l0_power = np.abs(w0).sum(axis=0)

    # Layer 1 (Hidden 256): incoming + outgoing weights
    l1_power = np.abs(w0).sum(axis=1) + np.abs(w2).sum(axis=0)

    # Layer 2 (Hidden 256): incoming + outgoing weights
    l2_power = np.abs(w2).sum(axis=1) + np.abs(w4).sum(axis=0)

    # Layer 3 (Hidden 128): incoming + outgoing weights
    l3_power = np.abs(w4).sum(axis=1) + np.abs(w6).sum(axis=0)

    # Layer 4 (Output 12): incoming weights
    l4_power = np.abs(w6).sum(axis=1)

    print("✅ Computed synaptic energy across all 701 neurons.")

    # Custom Glowing Bio-Neuro Colormap: Dark Navy -> Electric Blue -> Neon Cyan -> Bright Gold -> White
    colors = [(0.04, 0.06, 0.12), (0.08, 0.35, 0.75), (0.0, 0.85, 0.95), (0.95, 0.75, 0.1), (1.0, 1.0, 1.0)]
    bio_cmap = LinearSegmentedColormap.from_list("bio_glow", colors, N=256)

    # =========================================================================
    # IMAGE 1: FULL ARCHITECTURAL BRAIN MAP (NODE BRIGHTNESS = LEARNED WEIGHT)
    # =========================================================================
    print("🎨 Generating Image 1: Architectural Brain Weight Map...")

    fig, ax = plt.subplots(figsize=(24, 13.5), dpi=200, facecolor="#07090e")
    ax.set_facecolor("#07090e")

    # Layer X coordinates
    layer_x = [0.08, 0.30, 0.52, 0.73, 0.92]
    top_y, bot_y = 0.88, 0.08

    # Node positions
    coords = []

    # Layer 0: 49 nodes
    l0_coords = []
    for i in range(49):
        y = top_y - (i / 48) * (top_y - bot_y)
        l0_coords.append((layer_x[0], y))
    coords.append(l0_coords)

    # Layer 1: 256 nodes in 4 columns of 64
    l1_coords = []
    for i in range(256):
        c = i // 64
        r = i % 64
        x = layer_x[1] + (c - 1.5) * 0.022
        y = top_y - (r / 63) * (top_y - bot_y)
        l1_coords.append((x, y))
    coords.append(l1_coords)

    # Layer 2: 256 nodes in 4 columns of 64
    l2_coords = []
    for i in range(256):
        c = i // 64
        r = i % 64
        x = layer_x[2] + (c - 1.5) * 0.022
        y = top_y - (r / 63) * (top_y - bot_y)
        l2_coords.append((x, y))
    coords.append(l2_coords)

    # Layer 3: 128 nodes in 2 columns of 64
    l3_coords = []
    for i in range(128):
        c = i // 64
        r = i % 64
        x = layer_x[3] + (c - 0.5) * 0.026
        y = top_y - (r / 63) * (top_y - bot_y)
        l3_coords.append((x, y))
    coords.append(l3_coords)

    # Layer 4: 12 nodes
    l4_coords = []
    for i in range(12):
        y = top_y - 0.04 - (i / 11) * (top_y - bot_y - 0.08)
        l4_coords.append((layer_x[4], y))
    coords.append(l4_coords)

    # Draw Synaptic Pathways (strongest connections glow brightest)
    def draw_synapses(W, c1, c2, max_lines=250):
        flat_idx = np.argsort(np.abs(W).ravel())[-max_lines:]
        max_w = np.abs(W).max()
        for idx in flat_idx:
            out_i, in_j = np.unravel_index(idx, W.shape)
            p1 = c1[in_j]
            p2 = c2[out_i]
            val = abs(W[out_i, in_j]) / max_w
            alpha = min(0.45, 0.08 + val * 0.35)
            color = (0.0, 0.85, 0.95, alpha) if W[out_i, in_j] > 0 else (0.85, 0.3, 0.95, alpha)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=0.4 + val * 1.0, zorder=1)

    draw_synapses(w0, coords[0], coords[1], max_lines=200)
    draw_synapses(w2, coords[1], coords[2], max_lines=300)
    draw_synapses(w4, coords[2], coords[3], max_lines=250)
    draw_synapses(w6, coords[3], coords[4], max_lines=120)

    # Draw Nodes with Brightness & Glow Proportional to Learned Synaptic Weight
    all_powers = [l0_power, l1_power, l2_power, l3_power, l4_power]

    for l_idx, (layer_pts, powers) in enumerate(zip(coords, all_powers)):
        min_p = powers.min()
        max_p = powers.max()
        # Normalized power 0 to 1
        norm_p = (powers - min_p) / (max_p - min_p + 1e-8)

        base_radius = [24, 12, 12, 18, 55][l_idx]

        for i, (pt, p_val, norm_val) in enumerate(zip(layer_pts, powers, norm_p)):
            rgba = bio_cmap(norm_val)

            # Outer bloom halo for highly developed hub neurons
            if norm_val > 0.5:
                halo_size = base_radius * (1.8 + norm_val * 2.2)
                ax.scatter(pt[0], pt[1], s=halo_size * 4, color=rgba, alpha=0.15 * norm_val, edgecolors="none", zorder=2)
                ax.scatter(pt[0], pt[1], s=halo_size * 2, color=rgba, alpha=0.30 * norm_val, edgecolors="none", zorder=3)

            # Core neuron
            core_size = base_radius * (0.8 + norm_val * 1.5)
            ax.scatter(pt[0], pt[1], s=core_size, color=rgba, alpha=0.85 + norm_val * 0.15, edgecolors="#ffffff", linewidths=0.3, zorder=4)

    # Text Labels & Layer Annotations
    layer_names = [
        "INPUT LAYER (49)\nSensory Encoders",
        "HIDDEN LAYER 1 (256)\nFeature Extractors",
        "HIDDEN LAYER 2 (256)\nCore Associative Brain",
        "HIDDEN LAYER 3 (128)\nPremotor Motor Drivers",
        "OUTPUT LAYER (12)\nJoint Position Targets"
    ]

    for lx, name in zip(layer_x, layer_names):
        ax.text(lx, 0.94, name, color="#e2e8f0", fontsize=11, fontweight="bold", ha="center", va="center", family="sans-serif")

    # Output Joint Names
    joint_labels = [
        "L Hip Pitch", "L Hip Roll", "L Hip Yaw", "L Knee", "L Ankle Pitch", "L Ankle Roll",
        "R Hip Pitch", "R Hip Roll", "R Hip Yaw", "R Knee", "R Ankle Pitch", "R Ankle Roll"
    ]
    for i, (lbl, pt) in enumerate(zip(joint_labels, coords[4])):
        norm_w = (l4_power[i] - l4_power.min()) / (l4_power.max() - l4_power.min() + 1e-8)
        color = bio_cmap(norm_w)
        ax.text(pt[0] + 0.015, pt[1], f"{lbl} ({l4_power[i]:.1f})", color=color, fontsize=9.5, fontweight="semibold", va="center", family="monospace")

    # Title & Legend Box
    ax.text(0.04, 0.985, "UNITREE G1 — NEURAL NETWORK SYNAPTIC WEIGHT DEVELOPMENT MAP", color="#ffffff", fontsize=17, fontweight="bold", va="top")
    ax.text(0.04, 0.965, "Trained PPO Policy (5,500 Iterations)  |  Node Brightness Proportional to Total Synaptic Weight Magnitude (L1-Norm)", color="#00f2fe", fontsize=11, va="top", family="monospace")

    # Colorbar at bottom
    sm = plt.cm.ScalarMappable(cmap=bio_cmap, norm=plt.Normalize(vmin=0, vmax=100))
    cbar_ax = fig.add_axes([0.35, 0.025, 0.30, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 50, 100])
    cbar.set_ticklabels(["Dormant / Low Weight", "Moderate Synaptogenesis", "Highly Developed Brain Hub (Max Weight)"])
    cbar.ax.tick_params(labelsize=9, colors="#cbd5e1")
    cbar.outline.set_edgecolor("#334155")

    ax.set_xlim(0.02, 1.05)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    img1_path = Path("neural_network_brain_map.png")
    plt.tight_layout()
    plt.savefig(img1_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ Saved Image 1 to {img1_path.name}")

    # =========================================================================
    # IMAGE 2: NEURO-ANATOMICAL WEIGHT HEATMAPS & DEVELOPMENT DASHBOARD
    # =========================================================================
    print("🎨 Generating Image 2: Neuro-Anatomical Weight Heatmaps...")

    fig = plt.figure(figsize=(22, 13), dpi=200, facecolor="#080c14")
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.3, 1.3, 1.4], height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.28)

    # 1. W0 Heatmap: Input -> Layer 1 (256 x 49)
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(np.abs(w0), aspect="auto", cmap="inferno", interpolation="nearest")
    ax0.set_title("Layer 1 Receptive Field: $W_0$ (256 × 49)", color="#ffffff", fontsize=12, fontweight="bold", pad=8)
    ax0.set_xlabel("Input Sensor Channels (0 to 48)", color="#94a3b8", fontsize=9)
    ax0.set_ylabel("Hidden Neurons (0 to 255)", color="#94a3b8", fontsize=9)
    ax0.tick_params(colors="#94a3b8", labelsize=8)
    cb0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cb0.ax.tick_params(colors="#94a3b8", labelsize=8)

    # 2. W2 Heatmap: Core Brain Layer 1 -> Layer 2 (256 x 256)
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(np.abs(w2), aspect="auto", cmap="inferno", interpolation="nearest")
    ax1.set_title("Core Deep Matrix: $W_2$ (256 × 256)", color="#ffffff", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Layer 1 Neurons (0 to 255)", color="#94a3b8", fontsize=9)
    ax1.set_ylabel("Layer 2 Neurons (0 to 255)", color="#94a3b8", fontsize=9)
    ax1.tick_params(colors="#94a3b8", labelsize=8)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors="#94a3b8", labelsize=8)

    # 3. Sensory Feature Importance Bar Chart (Which parts of the body does the brain wire to?)
    ax2 = fig.add_subplot(gs[0, 2])
    categories = [
        "IMU Gyro (3)",
        "Proj. Gravity (3)",
        "Commands (3)",
        "Joint Pos (13)",
        "Joint Vel (13)",
        "Prev. Actions (12)",
        "Gait Clock (2)"
    ]
    mean_weights = [
        l0_power[0:3].mean(),
        l0_power[3:6].mean(),
        l0_power[6:9].mean(),
        l0_power[9:22].mean(),
        l0_power[22:35].mean(),
        l0_power[35:47].mean(),
        l0_power[47:49].mean()
    ]
    bar_colors = ["#00f2fe", "#38bdf8", "#818cf8", "#a855f7", "#ec4899", "#f43f5e", "#f59e0b"]
    bars = ax2.barh(categories, mean_weights, color=bar_colors, edgecolor="#ffffff", linewidth=0.5, height=0.65)
    ax2.set_title("Sensory Investment: Mean Weight per Channel", color="#ffffff", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Total Synaptic Connection Mass", color="#94a3b8", fontsize=9)
    ax2.tick_params(colors="#94a3b8", labelsize=9)
    ax2.set_facecolor("#0d131f")
    ax2.grid(axis="x", color="#1e293b", linestyle="--", alpha=0.7)
    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 0.6, bar.get_y() + bar.get_height()/2, f"{w:.1f}", color="#f1f5f9", va="center", fontsize=8.5, family="monospace")

    # 4. W4 Heatmap: Layer 2 -> Layer 3 (128 x 256)
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(np.abs(w4), aspect="auto", cmap="inferno", interpolation="nearest")
    ax3.set_title("Premotor Compression: $W_4$ (128 × 256)", color="#ffffff", fontsize=12, fontweight="bold", pad=8)
    ax3.set_xlabel("Layer 2 Neurons (0 to 255)", color="#94a3b8", fontsize=9)
    ax3.set_ylabel("Layer 3 Neurons (0 to 127)", color="#94a3b8", fontsize=9)
    ax3.tick_params(colors="#94a3b8", labelsize=8)
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(colors="#94a3b8", labelsize=8)

    # 5. W6 Heatmap: Motor Cortex Driving 12 Leg Joints (12 x 128)
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(np.abs(w6), aspect="auto", cmap="inferno", interpolation="nearest")
    ax4.set_title("Motor Cortex Driver Matrix: $W_6$ (12 × 128)", color="#ffffff", fontsize=12, fontweight="bold", pad=8)
    ax4.set_xlabel("Layer 3 Neurons (0 to 127)", color="#94a3b8", fontsize=9)
    ax4.set_yticks(range(12))
    ax4.set_yticklabels(joint_labels, fontsize=8, color="#cbd5e1", family="monospace")
    ax4.tick_params(colors="#94a3b8", labelsize=8)
    cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.ax.tick_params(colors="#94a3b8", labelsize=8)

    # 6. Neuron Synaptic Specialization Curve (Gini / Sparsity of Brain Development)
    ax5 = fig.add_subplot(gs[1, 2])
    sorted_l1 = np.sort(l1_power)
    sorted_l2 = np.sort(l2_power)
    sorted_l3 = np.sort(l3_power)
    ax5.plot(np.linspace(0, 100, len(sorted_l1)), sorted_l1, label="Layer 1 (256)", color="#00f2fe", lw=2)
    ax5.plot(np.linspace(0, 100, len(sorted_l2)), sorted_l2, label="Layer 2 (256)", color="#a855f7", lw=2)
    ax5.plot(np.linspace(0, 100, len(sorted_l3)), sorted_l3, label="Layer 3 (128)", color="#f59e0b", lw=2)
    ax5.set_title("Neuron Specialization: Synaptic Mass Distribution", color="#ffffff", fontsize=12, fontweight="bold", pad=8)
    ax5.set_xlabel("Neuron Percentile (%)", color="#94a3b8", fontsize=9)
    ax5.set_ylabel("Total Learned Synaptic Weight", color="#94a3b8", fontsize=9)
    ax5.set_facecolor("#0d131f")
    ax5.grid(color="#1e293b", linestyle="--", alpha=0.7)
    ax5.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9", fontsize=9)
    ax5.tick_params(colors="#94a3b8", labelsize=8.5)

    # Global Title
    fig.suptitle("UNITREE G1 HUMANOID — BRAIN SYNAPTIC WEIGHT ANALYSIS & FUNCTIONAL SPECIALIZATION", 
                 color="#ffffff", fontsize=16, fontweight="bold", y=0.98)

    img2_path = Path("neural_network_weight_heatmaps.png")
    plt.savefig(img2_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ Saved Image 2 to {img2_path.name}")

if __name__ == "__main__":
    generate_weight_visualizations()
