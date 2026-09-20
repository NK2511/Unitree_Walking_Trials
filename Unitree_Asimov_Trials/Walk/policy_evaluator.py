#!/usr/bin/env python3
"""Angad Robot Gait Evaluator.

This script evaluates a trained RL policy for the Angad biped robot.
It commands a fixed linear velocity (e.g. 1.0 m/s) over a user-defined duration,
logs raw telemetry at 200 Hz (physics rate), extracts stabilized gait cycles,
and plots publication-quality evaluation charts, including 3x2 joint analysis plots.
"""

import os
import sys
import time
import glob
import re
import argparse
import csv
from pathlib import Path

# Auto-relaunch inside the correct virtual environment if not already using it
VENV_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mjlab_env/bin/python"))
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    print(f"🔄 Auto-switching to mjlab_env Python interpreter...")
    os.execl(VENV_PYTHON, VENV_PYTHON, *sys.argv)

import sys
# Add current directory to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import tasks to populate registry
import mjlab.tasks  # noqa: F401
import config  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper


def find_latest_checkpoint(folder_path: str) -> str:
    path = Path(folder_path)
    if not path.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")
    
    # If the folder itself contains model files
    model_files = list(path.glob("model_*.pt"))
    
    # If not, look in subfolders (e.g. if logs directory was given directly)
    if not model_files:
        model_files = list(path.glob("**/model_*.pt"))
        
    if not model_files:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in '{folder_path}'.")
        
    # Extract iteration number and find the highest one
    valid_models = []
    for mf in model_files:
        match = re.search(r"model_(\d+)\.pt", mf.name)
        if match:
            valid_models.append((int(match.group(1)), mf))
            
    if not valid_models:
        raise FileNotFoundError(f"No valid checkpoint files matching 'model_*.pt' found.")
        
    valid_models.sort(key=lambda x: x[0])
    latest_path = valid_models[-1][1]
    return str(latest_path)


def zero_phase_lowpass(data, window_size=11):
    """Applies a double-sided moving average filter to ensure zero phase shift."""
    if len(data) <= window_size:
        return data
    window = np.ones(window_size) / window_size
    pad_width = window_size
    padded = np.pad(data, pad_width, mode='edge')
    # Forward pass
    forward = np.convolve(padded, window, mode='same')
    # Backward pass (time-reversed)
    backward = np.convolve(forward[::-1], window, mode='same')[::-1]
    # Strip padding
    return backward[pad_width:-pad_width]


# Global variables for patch access
current_vx_command = 1.0
physics_step_counter = 0
spawn_x = 0.0
spawn_y = 0.0
initialized_spawn = False
prev_left_contact = False
prev_right_contact = False
contact_threshold = 5.0  # Newtons

# Data logging lists (whole run)
times = []
com_x = []
com_y = []
desired_x = []
desired_y = []
actual_vx = []
cmd_vx = []

joint_names = []
joint_torques_history = {}
joint_positions_history = {}
joint_velocities_history = {}

left_foot_raw_forces = []
right_foot_raw_forces = []

left_steps_x = []
left_steps_y = []
right_steps_x = []
right_steps_y = []


def main():
    global current_vx_command, physics_step_counter, spawn_x, spawn_y, initialized_spawn
    global prev_left_contact, prev_right_contact, joint_names, joint_torques_history
    global joint_positions_history, joint_velocities_history, left_foot_raw_forces, right_foot_raw_forces

    parser = argparse.ArgumentParser(description="Angad Gait Evaluator")
    parser.add_argument("folder_path", type=str, help="Path to the training run logs folder")
    parser.add_argument("--task", type=str, default="Mjlab-Velocity-Flat-Angad", help="Task ID to load configs for")
    parser.add_argument("--speed", type=float, default=1.0, help="Commanded forward velocity in m/s")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation run duration in seconds")
    parser.add_argument("--ref-csv", type=str, default=None, help="Path to reference gait CSV for imitation tracking plot")
    parser.add_argument("--sweep", action="store_true", help="Sweep velocity from 0.0 to --speed over the duration")
    
    # Add parser support for render boolean flag
    parser.add_argument("--render", action="store_true", default=True, help="Render the GUI viewer (default)")
    parser.add_argument("--no-render", action="store_false", dest="render", help="Do not render the GUI viewer (headless)")
    
    args = parser.parse_args()

    # Set commanded velocity
    current_vx_command = 0.0 if args.sweep else args.speed
    if args.sweep:
        print(f"🎯 Evaluating with velocity sweep: 0.0 to {args.speed} m/s over {args.duration}s")
    else:
        print(f"🎯 Evaluating at fixed commanded velocity: {current_vx_command} m/s")

    # Resolve latest checkpoint
    checkpoint_path = find_latest_checkpoint(args.folder_path)
    print(f"✅ Selected latest checkpoint: {checkpoint_path}")

    # Set up directory for saving graphs and CSVs
    run_folder = Path(args.folder_path)
    folder_name = run_folder.name
    subfolder = "With_Torso" if "With_Torso" in args.folder_path else "No_Torso"
    output_dir = Path(__file__).resolve().parents[2] / "Model_Evaluations" / subfolder / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Saving outputs to: {output_dir}")

    # Load environment configuration
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)

    # Force single environment
    env_cfg.scene.num_envs = 1

    # Force Angad to spawn facing positive global X-axis (yaw = 0)
    if "reset_base" in env_cfg.events:
        env_cfg.events["reset_base"].params["pose_range"]["x"] = (0.0, 0.0)
        env_cfg.events["reset_base"].params["pose_range"]["y"] = (0.0, 0.0)
        env_cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)

    # Disable random starting offsets for robot joints to ensure determinism
    if "reset_robot_joints" in env_cfg.events:
        env_cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
        env_cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)

    # Set random seeds for reproducibility (e.g. deterministic gait clock phase)
    torch.manual_seed(42)
    np.random.seed(42)

    # Fix contact sensor secondary pattern to detect contact with terrain
    from mjlab.sensor.contact_sensor import ContactMatch
    for s_cfg in env_cfg.scene.sensors:
        if s_cfg.name == "feet_ground_contact":
            s_cfg.secondary = ContactMatch(mode="body", pattern="terrain")

    # Initialize environment
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy runner
    from dataclasses import asdict
    runner_cls = load_runner_cls(args.task) or OnPolicyRunner
    agent_dict = asdict(agent_cfg) if hasattr(agent_cfg, "__dict__") else agent_cfg
    if "policy" in agent_dict and "actor" not in agent_dict:
        policy_cfg = agent_dict.pop("policy")
        agent_dict["actor"] = {
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
        agent_dict["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("critic_hidden_dims", (128, 128, 128)),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("critic_obs_normalization", False),
        }
    if "obs_groups" in agent_dict:
        obs_groups = agent_dict["obs_groups"]
        if "policy" in obs_groups and "actor" not in obs_groups:
            obs_groups["actor"] = obs_groups.pop("policy")

    runner = runner_cls(env, agent_dict, device=device)
    
    # Patch torch.load to adapt older checkpoints on the fly
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        loaded = original_torch_load(*args, **kwargs)
        if isinstance(loaded, dict) and "model_state_dict" in loaded and "actor_state_dict" not in loaded:
            print("🔧 Adapting older checkpoint format ('model_state_dict') to new format ('actor_state_dict')...")
            model_sd = loaded["model_state_dict"]
            actor_sd = {}
            critic_sd = {}
            for k, v in model_sd.items():
                if k.startswith("actor."):
                    actor_sd[k.replace("actor.", "mlp.")] = v
                elif k.startswith("actor_obs_normalizer."):
                    actor_sd[k.replace("actor_obs_normalizer.", "obs_normalizer.")] = v
                elif k == "std":
                    actor_sd["distribution.std_param"] = v
                elif k.startswith("critic."):
                    critic_sd[k.replace("critic.", "mlp.")] = v
                elif k.startswith("critic_obs_normalizer."):
                    critic_sd[k.replace("critic_obs_normalizer.", "obs_normalizer.")] = v
            loaded["actor_state_dict"] = actor_sd
            loaded["critic_state_dict"] = critic_sd
        return loaded

    torch.load = patched_torch_load
    try:
        runner.load(checkpoint_path, map_location=device)
    finally:
        torch.load = original_torch_load
        
    policy = runner.get_inference_policy(device=device)

    # Setup command monkey-patching to control commanded velocities directly
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
    cmd_term = env.unwrapped.command_manager.get_term("twist")
    if isinstance(cmd_term, UniformVelocityCommand):
        def custom_resample_command(self, env_ids: torch.Tensor) -> None:
            pass  # Disable random resampling
            
        def custom_update_command(self) -> None:
            global current_vx_command
            self.vel_command_b[:, 0] = current_vx_command
            self.vel_command_b[:, 1] = 0.0
            self.vel_command_b[:, 2] = 0.0

        import types
        cmd_term._resample_command = types.MethodType(custom_resample_command, cmd_term)
        cmd_term._update_command = types.MethodType(custom_update_command, cmd_term)

    # Prepare joint names, positions, velocities, and torques log structures
    robot = env.unwrapped.scene["robot"]
    joint_names = robot.joint_names
    joint_torques_history = {name: [] for name in joint_names}
    joint_positions_history = {name: [] for name in joint_names}
    joint_velocities_history = {name: [] for name in joint_names}

    # Setup viewer if rendering is enabled
    viewer = None
    if args.render:
        print("[INFO] Launching passive MuJoCo viewer...")
        viewer = mujoco.viewer.launch_passive(
            env.unwrapped.sim.mj_model,
            env.unwrapped.sim.mj_data,
            show_left_ui=False,
            show_right_ui=False
        )

    # Patch sim.step to log values at 200 Hz
    original_sim_step = env.unwrapped.sim.step
    physics_dt = env.unwrapped.physics_dt
    contact_sensor = env.unwrapped.scene["feet_ground_contact"]

    # Debouncing variables for touchdown logging
    left_air_steps = 0
    right_air_steps = 0
    min_swing_steps = int(0.1 / physics_dt)  # 0.1s minimum in the air

    def patched_sim_step(self):
        global physics_step_counter, spawn_x, spawn_y, initialized_spawn
        global prev_left_contact, prev_right_contact, current_vx_command
        nonlocal left_air_steps, right_air_steps

        original_sim_step()

        # Get simulation data
        sim_data = env.unwrapped.sim.data

        # 1. Base Pos
        pos_w = robot.data.root_link_pos_w[0].cpu().numpy()
        x, y, z = pos_w[0], pos_w[1], pos_w[2]

        if not initialized_spawn:
            spawn_x = x
            spawn_y = y
            initialized_spawn = True

        t = physics_step_counter * physics_dt
        
        if args.sweep:
            current_vx_command = min(args.speed, (t / args.duration) * args.speed)

        # Log trajectory
        times.append(t)
        com_x.append(x)
        com_y.append(y)
        desired_x.append(x)  # Matches linear path progression
        desired_y.append(spawn_y)

        # Log velocities
        actual_vx_val = robot.data.root_link_lin_vel_b[0, 0].item()
        actual_vx.append(actual_vx_val)
        cmd_vx.append(current_vx_command)

        # Log joint angles (qpos index 7: covers joint hinges)
        qpos_vals = sim_data.qpos[0, 7:].cpu().numpy()
        for idx, name in enumerate(joint_names):
            joint_positions_history[name].append(qpos_vals[idx])

        # Log joint velocities (qvel index 6: covers joint hinges)
        qvel_vals = sim_data.qvel[0, 6:].cpu().numpy()
        for idx, name in enumerate(joint_names):
            joint_velocities_history[name].append(qvel_vals[idx])

        # Log joint torques (after the 6 floating-base DOFs)
        torques = sim_data.qfrc_actuator[0, 6:].cpu().numpy()
        for idx, name in enumerate(joint_names):
            joint_torques_history[name].append(torques[idx])

        # Log raw foot contact forces
        forces = contact_sensor.data.force[0].cpu().numpy()  # (2, 3)
        left_fz = abs(forces[0, 2])
        right_fz = abs(forces[1, 2])
        left_foot_raw_forces.append(left_fz)
        right_foot_raw_forces.append(right_fz)

        # Touchdown detection
        left_contact = left_fz > contact_threshold
        right_contact = right_fz > contact_threshold

        left_site_idx = robot.find_sites("left_foot")[0][0]
        right_site_idx = robot.find_sites("right_foot")[0][0]
        left_foot_pos = robot.data.site_pos_w[0, left_site_idx].cpu().numpy()
        right_foot_pos = robot.data.site_pos_w[0, right_site_idx].cpu().numpy()

        if left_contact:
            if left_air_steps >= min_swing_steps:
                left_steps_x.append(left_foot_pos[0])
                left_steps_y.append(left_foot_pos[1])
            left_air_steps = 0
        else:
            left_air_steps += 1

        if right_contact:
            if right_air_steps >= min_swing_steps:
                right_steps_x.append(right_foot_pos[0])
                right_steps_y.append(right_foot_pos[1])
            right_air_steps = 0
        else:
            right_air_steps += 1

        prev_left_contact = left_contact
        prev_right_contact = right_contact

        # Sync passive viewer if active
        if viewer is not None and viewer.is_running():
            mjd = env.unwrapped.sim.mj_data
            mjm = env.unwrapped.sim.mj_model
            mjd.qpos[:] = sim_data.qpos[0].cpu().numpy()
            mjd.qvel[:] = sim_data.qvel[0].cpu().numpy()
            mujoco.mj_forward(mjm, mjd)
            viewer.sync()
            time.sleep(physics_dt)

        physics_step_counter += 1

    import types
    env.unwrapped.sim.step = types.MethodType(patched_sim_step, env.unwrapped.sim)

    # Reset environment to initialize states and invoke events
    obs, info = env.reset()

    # Main evaluation loop (configured duration)
    dt = env.unwrapped.step_dt
    num_policy_steps = int(args.duration / dt)
    print(f"🚀 Starting simulation loop (Duration: {args.duration}s, Steps: {num_policy_steps})")

    for step in range(num_policy_steps):
        t_curr = step * dt

        with torch.no_grad():
            actions = policy(obs)

        obs, rewards, dones, info = env.step(actions)

        # Print progress overlay
        sys.stdout.write(f"\r⏳ Eval Progress: {t_curr:4.1f}s / {args.duration}s | Command: {current_vx_command:.2f} m/s")
        sys.stdout.flush()

        if dones.any():
            print(f"\n⚠️ Episode terminated early at t={t_curr:.2f}s due to environment signal.")
            break

    print("\n🏁 Simulation completed. Cleaning up and exporting data...")
    if viewer is not None:
        viewer.close()

    # ==========================================================
    # Gait Cycle Extraction (Touchdown to Touchdown of Left Foot)
    # ==========================================================
    # Look for left foot touchdowns after t >= 3.0s (to allow motion stabilization)
    touchdown_indices = []
    min_swing_steps_idx = int(0.1 / physics_dt)
    air_steps = 0
    for i in range(len(times)):
        left_contact = left_foot_raw_forces[i] > contact_threshold
        if not left_contact:
            air_steps += 1
        else:
            if air_steps >= min_swing_steps_idx:
                if times[i] >= 3.0:
                    touchdown_indices.append(i)
            air_steps = 0

    # 1. Slice for exactly 1 gait cycle (for foot forces and angles)
    if len(touchdown_indices) >= 2:
        one_cycle_start = touchdown_indices[0]
        one_cycle_end = touchdown_indices[1]
        one_cycle_slice = slice(one_cycle_start, one_cycle_end + 1)
        one_cycle_found = True
    else:
        print("⚠️ Could not detect a single clean gait cycle. Falling back to last 1.5s.")
        one_cycle_end = len(times) - 1
        one_cycle_start = max(0, len(times) - int(1.5 / physics_dt))
        one_cycle_slice = slice(one_cycle_start, one_cycle_end + 1)
        one_cycle_found = False

    # 2. Slice for exactly 4 gait cycles (for 3x2 joint analysis)
    if len(touchdown_indices) >= 5:
        multi_cycle_start = touchdown_indices[0]
        multi_cycle_end = touchdown_indices[4]
        multi_cycle_slice = slice(multi_cycle_start, multi_cycle_end + 1)
        num_cycles_plotted = 4
    elif len(touchdown_indices) >= 2:
        multi_cycle_start = touchdown_indices[0]
        multi_cycle_end = touchdown_indices[-1]
        multi_cycle_slice = slice(multi_cycle_start, multi_cycle_end + 1)
        num_cycles_plotted = len(touchdown_indices) - 1
    else:
        print(f"⚠️ Could not detect multi-gait cycles. Using full run duration ({times[-1]:.1f}s).")
        multi_cycle_slice = slice(0, len(times))
        num_cycles_plotted = f"Full Run ({times[-1]:.1f}s)"

    # Normalize 1-cycle time to 0-100%
    one_cycle_times = np.array(times[one_cycle_slice])
    if len(one_cycle_times) > 1:
        one_cycle_time_relative = one_cycle_times - one_cycle_times[0]
        one_cycle_duration = one_cycle_time_relative[-1]
        one_cycle_percent = (one_cycle_time_relative / one_cycle_duration) * 100.0
    else:
        one_cycle_percent = np.array([0.0])

    # Filter left and right joints dynamically
    left_joints = sorted([name for name in joint_names if name.startswith("left_")])
    right_joints = sorted([name for name in joint_names if name.startswith("right_")])

    # ==========================================================
    # CSV Data Export
    # ==========================================================
    
    # 1. COM Trajectory CSV
    with open(output_dir / "com_trajectory.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "com_x", "com_y", "desired_x", "desired_y"])
        for row in zip(times, com_x, com_y, desired_x, desired_y):
            writer.writerow(row)

    # 2. Joint Torques CSV (Full run)
    with open(output_dir / "joint_torques.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time"] + list(joint_names))
        for i in range(len(times)):
            row = [times[i]] + [joint_torques_history[name][i] for name in joint_names]
            writer.writerow(row)

    # 3. Foot Contact Forces CSV (Full run)
    with open(output_dir / "foot_forces_raw.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "left_foot_raw_force", "right_foot_raw_force"])
        for row in zip(times, left_foot_raw_forces, right_foot_raw_forces):
            writer.writerow(row)

    # 4. Velocity Tracking CSV (Full run)
    with open(output_dir / "velocity_tracking.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "cmd_vx", "actual_vx"])
        for row in zip(times, cmd_vx, actual_vx):
            writer.writerow(row)

    # 5. Footstep Positions CSV
    with open(output_dir / "footstep_positions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["foot", "x", "y"])
        for lx, ly in zip(left_steps_x, left_steps_y):
            writer.writerow(["left", lx, ly])
        for rx, ry in zip(right_steps_x, right_steps_y):
            writer.writerow(["right", rx, ry])

    print("📊 Saved CSV telemetry datasets.")

    # ==========================================================
    # Matplotlib Plot Generation
    # ==========================================================

    # Subplot styling colors (warm palette for left, cool for right)
    left_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2", "#8c564b", "#9467bd"]
    right_colors = ["#1f77b4", "#17becf", "#2ca02c", "#98df8a", "#aec7e8", "#393b79"]

    # Graph 1: Joint Angles for One Cycle (Left side vs Right side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, name in enumerate(left_joints):
        clean_label = name.replace("left_", "").replace("_", " ").title()
        ax1.plot(one_cycle_percent, np.array(joint_positions_history[name])[one_cycle_slice], 
                 label=clean_label, color=left_colors[idx % len(left_colors)], linewidth=2)
    ax1.set_title("Left Leg Joint Positions", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Gait Cycle (%)", fontsize=11)
    ax1.set_ylabel("Angle (rad)", fontsize=11)
    ax1.grid(True, linestyle=":")
    ax1.legend(loc="upper right", framealpha=0.9)

    for idx, name in enumerate(right_joints):
        clean_label = name.replace("right_", "").replace("_", " ").title()
        ax2.plot(one_cycle_percent, np.array(joint_positions_history[name])[one_cycle_slice], 
                 label=clean_label, color=right_colors[idx % len(right_colors)], linewidth=2)
    ax2.set_title("Right Leg Joint Positions", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Gait Cycle (%)", fontsize=11)
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.suptitle(f"Joint Angles for One Gait Cycle (Speed: {args.speed} m/s)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "joint_angles_cycle.png", dpi=200)
    plt.close()

    # Graph 2: 3x2 Joint Analysis plots (Torque, Velocity, and Torque vs Velocity Scatter) for each joint type
    joint_types = ["hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll"]
    
    # Extract times slice for multi-cycle
    multi_cycle_times = np.array(times[multi_cycle_slice])
    times_slice = multi_cycle_times - multi_cycle_times[0]

    for jt in joint_types:
        left_name = f"left_{jt}"
        right_name = f"right_{jt}"
        
        if left_name not in joint_names or right_name not in joint_names:
            continue

        # Extract raw data
        left_torque_raw = np.array(joint_torques_history[left_name])[multi_cycle_slice]
        right_torque_raw = np.array(joint_torques_history[right_name])[multi_cycle_slice]
        
        left_vel_raw = np.array(joint_velocities_history[left_name])[multi_cycle_slice]
        right_vel_raw = np.array(joint_velocities_history[right_name])[multi_cycle_slice]
        
        # Create figure
        fig, axs = plt.subplots(3, 2, figsize=(14, 15))
        jt_title = jt.replace("_", " ").title()
        
        # Row 1: Torque vs Time
        # Right Joint
        axs[0, 0].plot(times_slice, right_torque_raw, label="Torque", color="red", linewidth=2.0)
        axs[0, 0].set_title(f"Right {jt_title} Torque vs Time", fontsize=11, fontweight="bold")
        axs[0, 0].set_xlabel("Time (s)")
        axs[0, 0].set_ylabel("Torque (Nm)")
        axs[0, 0].grid(True, linestyle=":")
        axs[0, 0].legend()
        
        # Left Joint
        axs[0, 1].plot(times_slice, left_torque_raw, label="Torque", color="red", linewidth=2.0)
        axs[0, 1].set_title(f"Left {jt_title} Torque vs Time", fontsize=11, fontweight="bold")
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("Torque (Nm)")
        axs[0, 1].grid(True, linestyle=":")
        axs[0, 1].legend()

        # Row 2: Velocity vs Time
        # Right Joint
        axs[1, 0].plot(times_slice, right_vel_raw, label="Velocity", color="red", linewidth=2.0)
        axs[1, 0].set_title(f"Right {jt_title} Velocity vs Time", fontsize=11, fontweight="bold")
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("Velocity (rad/s)")
        axs[1, 0].grid(True, linestyle=":")
        axs[1, 0].legend()
        
        # Left Joint
        axs[1, 1].plot(times_slice, left_vel_raw, label="Velocity", color="red", linewidth=2.0)
        axs[1, 1].set_title(f"Left {jt_title} Velocity vs Time", fontsize=11, fontweight="bold")
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("Velocity (rad/s)")
        axs[1, 1].grid(True, linestyle=":")
        axs[1, 1].legend()

        # Row 3: Torque vs Velocity Scatter
        # Convert rad/s to RPM
        right_rotational_speed_rpm = np.abs(right_vel_raw) * (60.0 / (2.0 * np.pi))
        left_rotational_speed_rpm = np.abs(left_vel_raw) * (60.0 / (2.0 * np.pi))
        
        # Right Joint
        axs[2, 0].scatter(right_rotational_speed_rpm, np.abs(right_torque_raw), label="Data points", color="orange", s=8, alpha=0.8)
        axs[2, 0].set_title(f"Right {jt_title} |Torque| vs |Rotational Speed|", fontsize=11, fontweight="bold")
        axs[2, 0].set_xlabel("|Rotational Speed| (RPM)")
        axs[2, 0].set_ylabel("|Torque| (Nm)")
        axs[2, 0].grid(True, linestyle=":")
        axs[2, 0].legend()
        
        # Left Joint
        axs[2, 1].scatter(left_rotational_speed_rpm, np.abs(left_torque_raw), label="Data points", color="orange", s=8, alpha=0.8)
        axs[2, 1].set_title(f"Left {jt_title} |Torque| vs |Rotational Speed|", fontsize=11, fontweight="bold")
        axs[2, 1].set_xlabel("|Rotational Speed| (RPM)")
        axs[2, 1].set_ylabel("|Torque| (Nm)")
        axs[2, 1].grid(True, linestyle=":")
        axs[2, 1].legend()

        plt.suptitle(f"Walking at {args.speed} m/s ({num_cycles_plotted} cycles): joint Data for Right {jt_title} and Left {jt_title}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / f"joint_analysis_{jt}.png", dpi=200)
        plt.close()
        print(f"📈 Generated 3x2 joint analysis plot for: {jt_title}")

    # Graph 3: Forward Velocity tracking (Whole run)
    plt.figure(figsize=(10, 6))
    plt.plot(times, cmd_vx, label="Commanded Vx", color="black", linestyle="--", linewidth=2.0)
    plt.plot(times, actual_vx, label="Actual Vx", color="#2ca02c", alpha=0.9, linewidth=1.5)
    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Forward Speed (m/s)", fontsize=11)
    plt.title("Linear Forward Velocity Command Tracking", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_dir / "forward_velocity_tracking.png", dpi=200)
    plt.close()

    # Graph 4: Unfiltered Contact Force (Impulse forces) for One Cycle
    plt.figure(figsize=(10, 6))
    plt.plot(one_cycle_percent, np.array(left_foot_raw_forces)[one_cycle_slice], 
             label="Left Foot Force", color="#d62728", alpha=0.9, linewidth=1.8)
    plt.plot(one_cycle_percent, np.array(right_foot_raw_forces)[one_cycle_slice], 
             label="Right Foot Force", color="#1f77b4", alpha=0.9, linewidth=1.8)
    plt.xlabel("Gait Cycle (%)", fontsize=11)
    plt.ylabel("Vertical Contact Force (N)", fontsize=11)
    plt.title(f"Unfiltered Foot Ground Contact Forces (Impulses) for One Cycle (Speed: {args.speed} m/s)", 
              fontsize=13, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_dir / "foot_forces_cycle.png", dpi=200)
    plt.close()

    # Graph 5: Trajectory COM & Footstep Positions
    plt.figure(figsize=(10, 6))
    plt.plot(com_x, com_y, label="Actual COM Path", color="purple", linewidth=2)
    plt.plot(desired_x, desired_y, label="Desired Straight Path", color="black", linestyle="--", alpha=0.7)
    if left_steps_x:
        plt.scatter(left_steps_x, left_steps_y, color="red", label="Left Footsteps", marker="o", s=50, edgecolors="darkred")
    if right_steps_x:
        plt.scatter(right_steps_x, right_steps_y, color="blue", label="Right Footsteps", marker="s", s=50, edgecolors="darkblue")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Angad COM Path Trajectory & Footstep Positions")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_dir / "com_trajectory_and_steps.png", dpi=200)
    plt.close()

    # Graph 6: Continuous Tracking Error Analysis (over middle ~4 seconds)
    ref_csv_path = args.ref_csv
    if not ref_csv_path:
        # Fallback to default
        import unitree_constants
        ref_csv_path = str(Path(__file__).resolve().parents[1] / "Gaits" / "csvs" / "unitree_g1_speed_mapped_gait.csv")
    
    if os.path.exists(ref_csv_path) and ref_csv_path.endswith(".csv") and ref_csv_path != "none":
        print(f"🔄 Plotting Imitation Tracking Error against: {ref_csv_path}")
        
        ref_data = {}
        with open(ref_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for col in reader.fieldnames:
                ref_data[col] = []
            for row_dict in reader:
                for col in reader.fieldnames:
                    ref_data[col].append(float(row_dict[col]))
                    
        # Map CSV columns to MuJoCo joint names
        csv_to_mujoco_map = {
            "left_hip_pitch": "left_hip_pitch", "left_hip_roll": "left_hip_roll", "left_hip_yaw": "left_hip_yaw",
            "left_knee": "left_knee_pitch", "left_ankle_pitch": "left_ankle_pitch", "left_ankle_roll": "left_ankle_roll",
            "right_hip_pitch": "right_hip_pitch", "right_hip_roll": "right_hip_roll", "right_hip_yaw": "right_hip_yaw",
            "right_knee": "right_knee_pitch", "right_ankle_pitch": "right_ankle_pitch", "right_ankle_roll": "right_ankle_roll",
        }
        
        if len(touchdown_indices) >= 2:
            # Create a continuous desired trajectory by stretching the CSV between every touchdown
            continuous_desired = {col: [] for col in csv_to_mujoco_map.values()}
            continuous_actual = {col: [] for col in csv_to_mujoco_map.values()}
            continuous_times = []
            
            # Find the best phase shift using cross-correlation on Left Hip Pitch
            # We assume Left Hip Pitch exists in both
            ref_lhp = np.array(ref_data["left_hip_pitch"])
            first_cycle_len = touchdown_indices[1] - touchdown_indices[0]
            act_lhp = np.array(joint_positions_history["left_hip_pitch"])[touchdown_indices[0]:touchdown_indices[1]]
            ref_lhp_interp = np.interp(np.linspace(0, 1, first_cycle_len), np.linspace(0, 1, len(ref_lhp)), ref_lhp)
            
            # Cross-correlate to find phase offset
            correlation = np.correlate(act_lhp - np.mean(act_lhp), np.tile(ref_lhp_interp, 2) - np.mean(ref_lhp_interp), mode='valid')
            best_shift = np.argmax(correlation)
            
            # Limit to roughly middle 4 seconds (e.g. from t=3 to t=7)
            # Find touchdowns within this window
            valid_tds = [td for td in touchdown_indices if 3.0 <= times[td] <= 7.0]
            if len(valid_tds) < 2:
                valid_tds = touchdown_indices  # Fallback to all available if window is too small
                
            for i in range(len(valid_tds) - 1):
                start_idx = valid_tds[i]
                end_idx = valid_tds[i+1]
                cycle_len = end_idx - start_idx
                continuous_times.extend(times[start_idx:end_idx])
                
                for csv_col, mj_col in csv_to_mujoco_map.items():
                    if csv_col in ref_data and mj_col in joint_positions_history:
                        ref_arr = np.array(ref_data[csv_col])
                        # Apply phase shift
                        shifted_ref = np.roll(ref_arr, int(best_shift * (len(ref_arr) / first_cycle_len)))
                        
                        # Interpolate to cycle length
                        interp_desired = np.interp(np.linspace(0, 1, cycle_len), np.linspace(0, 1, len(shifted_ref)), shifted_ref)
                        
                        continuous_desired[mj_col].extend(interp_desired)
                        continuous_actual[mj_col].extend(joint_positions_history[mj_col][start_idx:end_idx])
                        
            fig, axs = plt.subplots(6, 2, figsize=(16, 20), sharex=True)
            joint_categories = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
            
            for row, cat in enumerate(joint_categories):
                for col, side in enumerate(["left", "right"]):
                    csv_col = f"{side}_{cat}"
                    if csv_col not in ref_data:
                        continue
                    mujoco_joint = csv_to_mujoco_map[csv_col]
                    if mujoco_joint not in continuous_actual or len(continuous_actual[mujoco_joint]) == 0:
                        continue
                    
                    actual_arr = np.array(continuous_actual[mujoco_joint])
                    desired_arr = np.array(continuous_desired[mujoco_joint])
                    error_arr = actual_arr - desired_arr
                    
                    axs[row, col].plot(continuous_times, error_arr, label="Error (Actual - Desired)", color="purple", linewidth=2.0)
                    axs[row, col].axhline(0, color="black", linestyle="--", alpha=0.5)
                    
                    title = f"{side.title()} {cat.replace('_', ' ').title()}"
                    axs[row, col].set_title(title, fontsize=12, fontweight="bold")
                    axs[row, col].set_ylabel("Error (rad)")
                    axs[row, col].grid(True, linestyle=":")
                    axs[row, col].legend()
                    
                    if row == 5:
                        axs[row, col].set_xlabel("Time (s)", fontsize=11)
            
            plt.suptitle(f"Continuous Tracking Error Analysis (Middle 4s Window)\nCommand: {args.speed} m/s", fontsize=16, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.savefig(output_dir / "imitation_tracking_error.png", dpi=200)
            plt.close()
            print("📈 Exported Imitation Tracking Error plot.")
        else:
            print("⚠️ Not enough gait cycles detected to plot continuous tracking error.")
    else:
        print(f"⚠️ Could not find reference CSV at {ref_csv_path}. Skipping Imitation Tracking Error plot.")

    print("📈 Exported evaluation performance plots. Done!")


if __name__ == "__main__":
    main()
