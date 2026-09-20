"""
plot_policy_diagnostics.py
===========================
Runs a policy rollout for 500 steps, records:
  - 2D XY Trajectory of Pelvis vs Footstep Targets
  - Robot Base Heading (Yaw) vs Target Heading over Time
  - Active Foot-to-Target Distance over Time
  - Left & Right Foot Z-Heights (Swing Arcs) over Time

Saves all 4 plots into a clean 2x2 PNG image grid: 'policy_diagnostic_plots.png'
which can be provided to inspect the robot's physical behavior visually.
"""

import os
import sys
import glob
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

VENV_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mjlab_env/bin/python"))
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    os.execl(VENV_PYTHON, VENV_PYTHON, *sys.argv)

import config
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from rsl_rl.runners import OnPolicyRunner
from dataclasses import asdict

def quat_to_yaw(qw, qx, qy, qz):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)

import math

def run_diagnostics_and_plot(max_steps: int = 500, output_png: str = "policy_diagnostic_plots.png"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    task_id = "Mjlab-Footstep-Flat-Angad"

    log_root = Path(__file__).parent / "logs" / "rsl_rl"
    model_files = glob.glob(os.path.join(log_root, "**", "model_*.pt"), recursive=True)
    
    latest_ckpt = None
    if model_files:
        model_files.sort(key=lambda x: os.path.getmtime(x))
        latest_ckpt = model_files[-1]
        print(f"🔍 Found checkpoint: {latest_ckpt}")

    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    from config.footstep_env_cfg import _patch_env_with_footstep_manager
    _patch_env_with_footstep_manager(env)
    wrapped_env = RslRlVecEnvWrapper(env)

    policy = None
    if latest_ckpt:
        try:
            agent_cfg = load_rl_cfg(task_id)
            agent_cfg_dict = asdict(agent_cfg) if hasattr(agent_cfg, '__dataclass_fields__') else agent_cfg
            
            if "policy" in agent_cfg_dict and "actor" not in agent_cfg_dict:
                policy_cfg = agent_cfg_dict.pop("policy")
                agent_cfg_dict["actor"] = {
                    "class_name": "MLPModel",
                    "hidden_dims": policy_cfg.get("actor_hidden_dims", (128, 128, 128)),
                    "activation": policy_cfg.get("activation", "elu"),
                    "obs_normalization": policy_cfg.get("actor_obs_normalization", False),
                    "distribution_cfg": {
                        "class_name": "GaussianDistribution",
                        "init_std": policy_cfg.get("init_noise_std", 1.0),
                        "std_type": policy_cfg.get("noise_std_type", "scalar"),
                    }
                }
                agent_cfg_dict["critic"] = {
                    "class_name": "MLPModel",
                    "hidden_dims": policy_cfg.get("critic_hidden_dims", (128, 128, 128)),
                    "activation": policy_cfg.get("activation", "elu"),
                    "obs_normalization": policy_cfg.get("critic_obs_normalization", False),
                }
            if "obs_groups" in agent_cfg_dict:
                obs_groups = agent_cfg_dict["obs_groups"]
                if "policy" in obs_groups and "actor" not in obs_groups:
                    obs_groups["actor"] = obs_groups.pop("policy")

            # Inspect state dict to check input feature dimension (51-D vs 53-D)
            import torch.nn as nn
            ckpt_data = torch.load(str(latest_ckpt), map_location=device)
            in_dim = 53
            if "actor_state_dict" in ckpt_data and "mlp.0.weight" in ckpt_data["actor_state_dict"]:
                in_dim = ckpt_data["actor_state_dict"]["mlp.0.weight"].shape[1]
            elif "model_state_dict" in ckpt_data:
                for k, v in ckpt_data["model_state_dict"].items():
                    if "actor.mlp.0.weight" in k or "mlp.0.weight" in k:
                        in_dim = v.shape[1]
                        break
            
            print(f"📊 Checkpoint expects input dimension: {in_dim}")

            runner = OnPolicyRunner(wrapped_env, agent_cfg_dict, device=device)

            if in_dim == 51:
                # Replace linear layer in actor & critic to match legacy 51-D / 63-D shapes
                runner.alg.actor.mlp[0] = nn.Linear(51, 512).to(device)
                runner.alg.critic.mlp[0] = nn.Linear(63, 512).to(device)
                runner.load(str(latest_ckpt), map_location=device)
                raw_policy = runner.get_inference_policy(device=device)
                
                def policy(obs_input):
                    td = obs_input.clone() if hasattr(obs_input, "clone") else obs_input
                    obs_53 = td["policy"] if isinstance(td, dict) or hasattr(td, "__getitem__") else td
                    obs_51 = torch.cat([obs_53[:, :45], obs_53[:, 46:50], obs_53[:, 51:]], dim=1)
                    td["policy"] = obs_51
                    return raw_policy(td)
                
                print("✅ Successfully loaded legacy 51-D policy with observation adapter!")
            else:
                runner.load(str(latest_ckpt), map_location=device)
                policy = runner.get_inference_policy(device=device)
                print("✅ Successfully loaded 53-D policy!")
        except Exception as e:
            print(f"⚠️ Could not load policy ({e}). Running baseline zero policy.")

    obs_dict, _ = wrapped_env.reset()
    robot = env.scene["robot"]
    fm = env.footstep_manager
    site_ids, _ = robot.find_sites(["left_foot_site", "right_foot_site"])

    # Logging arrays over time
    times = []
    pelvis_xy = []
    left_foot_xyz = []
    right_foot_xyz = []
    robot_yaws = []
    target_yaws = []
    active_foot_dists = []
    target_hit_flags = []

    dt = env.step_dt
    
    # Store initial target sequence
    n_seq = fm.num_active_steps[0].item()
    target_sequence_w = fm.sequence[0, :n_seq].cpu().numpy()  # [N, 4]

    for step_i in range(max_steps):
        t_sec = step_i * dt
        
        with torch.inference_mode():
            if policy is not None:
                obs_tensor = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
                actions = policy(obs_tensor)
            else:
                actions = torch.zeros((1, env.action_manager.total_action_dim), device=device)

        obs_dict, rew, dones, extras = wrapped_env.step(actions)

        # Record physical data
        pelv_pos = robot.data.root_link_pos_w[0].cpu().numpy()
        pelvis_xy.append(pelv_pos[:2])

        l_pos = robot.data.site_pos_w[0, site_ids[0]].cpu().numpy()
        r_pos = robot.data.site_pos_w[0, site_ids[1]].cpu().numpy()
        left_foot_xyz.append(l_pos)
        right_foot_xyz.append(r_pos)

        quat = robot.data.root_link_quat_w[0].cpu().numpy()
        r_yaw = math.degrees(quat_to_yaw(quat[0], quat[1], quat[2], quat[3]))
        robot_yaws.append(r_yaw)

        t1_w = fm.get_target_pos_world()[0].cpu().numpy()
        t1_full = fm._gather_target(fm.t1)[0].cpu().numpy()
        t1_yaw_deg = math.degrees(t1_full[3])
        target_yaws.append(t1_yaw_deg)

        t1_is_left = fm.get_active_target_is_left()[0].item()
        dist = np.linalg.norm(l_pos - t1_w) if t1_is_left else np.linalg.norm(r_pos - t1_w)
        active_foot_dists.append(dist)

        hit = (dist < fm.target_radius)
        target_hit_flags.append(hit)
        times.append(t_sec)

    # Convert to numpy arrays
    times = np.array(times)
    pelvis_xy = np.array(pelvis_xy)
    left_foot_xyz = np.array(left_foot_xyz)
    right_foot_xyz = np.array(right_foot_xyz)
    robot_yaws = np.array(robot_yaws)
    target_yaws = np.array(target_yaws)
    active_foot_dists = np.array(active_foot_dists)

    # ─── GENERATE 4-PANEL DIAGNOSTIC GRAPH ────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Angad Footstep RL Policy Diagnostic Report", fontsize=16, fontweight='bold')

    # Panel 1: 2D Trajectory (Pelvis vs Footstep Targets)
    ax1 = axs[0, 0]
    ax1.plot(pelvis_xy[:, 0], pelvis_xy[:, 1], 'g-', linewidth=2, label='Pelvis Trajectory')
    ax1.plot(pelvis_xy[0, 0], pelvis_xy[0, 1], 'go', markersize=10, label='Spawn Start')
    
    for i in range(len(target_sequence_w)):
        tx, ty, _, _ = target_sequence_w[i]
        is_l = (i % 2 == 0)
        c = 'blue' if is_l else 'red'
        ax1.scatter(tx, ty, color=c, s=80, alpha=0.8, edgecolors='black')
        ax1.text(tx+0.03, ty+0.03, f"{i}", fontsize=8)
    
    ax1.scatter([], [], color='blue', label='Left Target')
    ax1.scatter([], [], color='red', label='Right Target')
    ax1.set_title("1. 2D Trajectory: Pelvis Path vs Target Footsteps")
    ax1.set_xlabel("World X (m)")
    ax1.set_ylabel("World Y (m)")
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axis('equal')

    # Panel 2: Robot Heading (Yaw) vs Target Heading over Time
    ax2 = axs[0, 1]
    ax2.plot(times, robot_yaws, 'b-', label='Robot Yaw (deg)')
    ax2.plot(times, target_yaws, 'r--', label='Target t1 Yaw (deg)')
    ax2.set_title("2. Heading Alignment: Robot Yaw vs Target Yaw over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Heading Angle (deg)")
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Panel 3: Active Foot-to-Target Distance over Time
    ax3 = axs[1, 0]
    ax3.plot(times, active_foot_dists, 'm-', label='Active Foot -> Target Dist (m)')
    ax3.axhline(y=fm.target_radius, color='g', linestyle='--', label=f'Target Radius ({fm.target_radius}m)')
    ax3.set_title("3. Active Foot Target Distance over Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Distance (m)")
    ax3.legend(loc='best')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Panel 4: Foot Z-Heights (Swing Arcs) over Time
    ax4 = axs[1, 1]
    ax4.plot(times, left_foot_xyz[:, 2], 'b-', alpha=0.8, label='Left Foot Z (m)')
    ax4.plot(times, right_foot_xyz[:, 2], 'r-', alpha=0.8, label='Right Foot Z (m)')
    ax4.axhline(y=0.0644, color='k', linestyle=':', label='Ground Level')
    ax4.set_title("4. Foot Heights (Z): Swing Arcs & Ground Contact")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Foot Z Height (m)")
    ax4.legend(loc='best')
    ax4.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_png, dpi=150)
    plt.close()

    print(f"\n📊 Diagnostic plots successfully generated: '{output_png}'")
    print(f"   Summary stats:")
    print(f"     - Total steps simulated : {max_steps} ({max_steps*dt:.1f}s)")
    print(f"     - Final pelvis position  : ({pelvis_xy[-1, 0]:.2f}, {pelvis_xy[-1, 1]:.2f}) m")
    print(f"     - Final robot yaw        : {robot_yaws[-1]:.1f}°")
    print(f"     - Mean foot target dist  : {active_foot_dists.mean():.3f} m")
    print(f"     - Foot target hits       : {sum(target_hit_flags)} frames")

    env.close()

if __name__ == "__main__":
    run_diagnostics_and_plot()
