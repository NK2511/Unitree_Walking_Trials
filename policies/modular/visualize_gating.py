"""Visualization & Telemetry Tool for Modular Unitree Brain.

This script demonstrates the dynamic interpretability of the modular brain
by simulating different real-world scenarios and plotting how the network's
attention shifts in real-time between the Balance, Gait, and Command sub-brains.

Run with:
  python visualize_gating.py
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from modular_architecture import ModularActor


def simulate_gating_telemetry(save_path: str = "modular_gating_telemetry.png"):
    print("Simulating 300 steps of locomotion under external disturbances...")

    actor = ModularActor(obs_dim=49, num_actions=12)
    actor.eval()

    steps = 300
    t = np.linspace(0, 10, steps)

    # Simulated sensory streams:
    # 0 to 100: Quiet steady forward walking
    # 100 to 160: External push disturbance (sudden torso pitch/roll & high gyro)
    # 160 to 220: Recovery & rapid directional steering change
    # 220 to 300: High speed sprint

    history_weights = []

    for i in range(steps):
        obs = torch.zeros(1, 49)

        if i < 100:
            # Steady walking
            obs[0, 6:9] = torch.tensor([0.0, 0.0, -1.0])
            obs[0, 9:12] = torch.tensor([0.8, 0.0, 0.0])  # 0.8 m/s forward
            obs[0, 48] = np.sin(2 * np.pi * t[i] * 1.5)
        elif 100 <= i < 160:
            # Severe push disturbance
            obs[0, 3:6] = torch.randn(3) * 2.0  # violent rotation
            obs[0, 6:9] = torch.tensor([0.7, -0.5, -0.5])  # tilted 45 degrees
            obs[0, 9:12] = torch.tensor([0.0, 0.0, 0.0])
            obs[0, 48] = 0.0
        elif 160 <= i < 220:
            # Steering / turning
            obs[0, 6:9] = torch.tensor([0.1, 0.0, -0.98])
            obs[0, 9:12] = torch.tensor([0.5, 0.0, 1.2])  # high yaw rate command
            obs[0, 48] = np.sin(2 * np.pi * t[i] * 2.0)
        else:
            # Fast sprint
            obs[0, 6:9] = torch.tensor([0.05, 0.0, -0.99])
            obs[0, 9:12] = torch.tensor([2.0, 0.0, 0.0])  # 2.0 m/s sprint
            obs[0, 48] = np.sin(2 * np.pi * t[i] * 2.5)

        with torch.no_grad():
            _, weights, _ = actor.forward_experts(obs)
            history_weights.append(weights.squeeze(0).cpu().numpy())

    history_weights = np.array(history_weights)  # (300, 3)

    # Plotting
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(t, history_weights[:, 0] * 100, label="Balance & Posture Expert (IMU/Tilt)", color="#00f2fe", lw=2.2)
    ax1.plot(t, history_weights[:, 1] * 100, label="Gait & Stepping Expert (Phase/Joints)", color="#a855f7", lw=2.2)
    ax1.plot(t, history_weights[:, 2] * 100, label="Command Tracking Expert (Joystick)", color="#10b981", lw=2.2)

    # Annotations of physical events
    ax1.axvspan(t[100], t[159], color="red", alpha=0.15, label="External Push Disturbance")
    ax1.axvspan(t[160], t[219], color="cyan", alpha=0.12, label="Steering Adjustment")
    ax1.axvspan(t[220], t[299], color="orange", alpha=0.12, label="High-Speed Sprint")

    ax1.set_ylabel("Expert Allocation Weight (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Unitree G1 Modular Brain — Real-Time Sub-Network Attention Telemetry", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right", framealpha=0.8)

    # Stacked percentage area in bottom plot
    ax2.stackplot(t, history_weights[:, 0]*100, history_weights[:, 1]*100, history_weights[:, 2]*100,
                  labels=["Balance", "Gait", "Command"], colors=["#00f2fe", "#a855f7", "#10b981"], alpha=0.7)
    ax2.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Total Share", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, t[-1])

    out_file = os.path.join(os.path.dirname(__file__), save_path)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"\n[OK] Telemetry plot saved successfully to:\n  {out_file}")


if __name__ == "__main__":
    simulate_gating_telemetry()
