#!/usr/bin/env python3
"""Angad Robot Video Generator.

This script loads a trained RL policy for the Angad biped robot, commands
a linear velocity ramp from 0.18 to 3.0 m/s over 15 seconds, and records
three MP4 videos (Side, Front, Isometric) using headless MuJoCo rendering.
"""

import os
import sys
import time
import re
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import mujoco

# Auto-relaunch inside the correct virtual environment if not already using it
VENV_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mjlab_env/bin/python"))
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    print(f"🔄 Auto-switching to mjlab_env Python interpreter...")
    os.execl(VENV_PYTHON, VENV_PYTHON, *sys.argv)

# Add current directory to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import mjlab.tasks  # noqa: F401
import config  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

current_vx_command = 0.0

def find_latest_checkpoint(folder_path: str) -> str:
    path = Path(folder_path)
    if not path.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")
    model_files = list(path.glob("model_*.pt"))
    if not model_files:
        model_files = list(path.glob("**/model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in '{folder_path}'.")
        
    valid_models = []
    for mf in model_files:
        match = re.search(r"model_(\d+)\.pt", mf.name)
        if match:
            valid_models.append((int(match.group(1)), mf))
            
    if not valid_models:
        raise FileNotFoundError("No valid checkpoint files matching 'model_*.pt' found.")
        
    valid_models.sort(key=lambda x: x[0])
    latest_path = valid_models[-1][1]
    return str(latest_path)

def setup_env(task, checkpoint_path):
    env_cfg = load_env_cfg(task, play=True)
    agent_cfg = load_rl_cfg(task)
    env_cfg.scene.num_envs = 1

    # Force spawn determinism
    if "reset_base" in env_cfg.events:
        env_cfg.events["reset_base"].params["pose_range"]["x"] = (0.0, 0.0)
        env_cfg.events["reset_base"].params["pose_range"]["y"] = (0.0, 0.0)
        env_cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
    if "reset_robot_joints" in env_cfg.events:
        env_cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
        env_cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)

    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task) or OnPolicyRunner
    from dataclasses import asdict
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
    
    # Auto-patch legacy rsl_rl checkpoints to the new format during load
    import unittest.mock
    original_load = torch.load
    def patched_torch_load(*args, **kwargs):
        loaded_dict = original_load(*args, **kwargs)
        if isinstance(loaded_dict, dict) and "actor_state_dict" not in loaded_dict and "model_state_dict" in loaded_dict:
            print("\n🔧 Auto-patching legacy checkpoint format (model_state_dict -> actor_state_dict)")
            loaded_dict["actor_state_dict"] = {}
            loaded_dict["critic_state_dict"] = {}
            for k, v in loaded_dict["model_state_dict"].items():
                if k.startswith("actor."):
                    loaded_dict["actor_state_dict"][k.replace("actor.", "mlp.", 1)] = v
                elif k.startswith("critic."):
                    loaded_dict["critic_state_dict"][k.replace("critic.", "mlp.", 1)] = v
                elif k == "distribution.std" or k == "std":
                    loaded_dict["actor_state_dict"]["distribution.std_param"] = v
                elif k.startswith("actor_obs_normalizer."):
                    loaded_dict["actor_state_dict"][k.replace("actor_obs_normalizer.", "obs_normalizer.", 1)] = v
                elif k.startswith("critic_obs_normalizer."):
                    loaded_dict["critic_state_dict"][k.replace("critic_obs_normalizer.", "obs_normalizer.", 1)] = v
                else:
                    loaded_dict["actor_state_dict"][k] = v
        return loaded_dict

    with unittest.mock.patch("torch.load", side_effect=patched_torch_load):
        runner.load(checkpoint_path, map_location=device)
        
    policy = runner.get_inference_policy(device=device)

    # Patch command manager
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
    cmd_term = env.unwrapped.command_manager.get_term("twist")
    if isinstance(cmd_term, UniformVelocityCommand):
        def custom_resample_command(self, env_ids: torch.Tensor) -> None:
            pass
        def custom_update_command(self) -> None:
            global current_vx_command
            self.vel_command_b[:, 0] = current_vx_command
            self.vel_command_b[:, 1] = 0.0
            self.vel_command_b[:, 2] = 0.0

        import types
        cmd_term._resample_command = types.MethodType(custom_resample_command, cmd_term)
        cmd_term._update_command = types.MethodType(custom_update_command, cmd_term)

    return env, policy

def record_video(env, policy, output_path, cam_config):
    print(f"🎥 Recording: {output_path.name}")
    
    mj_model = env.unwrapped.sim.mj_model
    mj_data = env.unwrapped.sim.mj_data
    renderer = mujoco.Renderer(mj_model, 720, 1280)
    
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = cam_config['distance']
    cam.azimuth = cam_config['azimuth']
    cam.elevation = cam_config['elevation']
    
    # Safely find the base body ID (handling mjlab prefixing like 'robot/base')
    base_id = -1
    for i in range(mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and ("base" in name or "pelvis" in name):
            base_id = i
            break
            
    if base_id != -1:
        cam.trackbodyid = base_id
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    
    import imageio
    writer = imageio.get_writer(str(output_path), fps=50, macro_block_size=None)
    
    global current_vx_command
    obs, _ = env.reset()
    robot_entity = env.unwrapped.scene["robot"]
    
    duration = 15.0
    dt = env.unwrapped.step_dt
    steps = int(duration / dt)
    
    for step in range(steps):
        t_curr = step * dt
        current_vx_command = 0.18 + (t_curr / duration) * (3.0 - 0.18)
        
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, info = env.step(actions)
        
        # Update camera lookat to robot's root position on the GPU
        root_pos = robot_entity.data.root_link_pos_w[0].cpu().numpy()
        if cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
            cam.lookat[:] = root_pos
            
        # CRITICAL FIX: The physics runs on the GPU, but the renderer reads from the CPU mj_data.
        # We must copy the current frame's joint positions from the GPU to the CPU!
        sim_data = env.unwrapped.sim.data
        mj_data.qpos[:] = sim_data.qpos[0].cpu().numpy()
        mj_data.qvel[:] = sim_data.qvel[0].cpu().numpy()
        mujoco.mj_forward(mj_model, mj_data)
        
        renderer.update_scene(mj_data, camera=cam)
        pixels = renderer.render()
        
        # Overlay text using cv2
        import cv2
        pixels = cv2.putText(
            pixels.copy(), 
            f"Command Speed: {current_vx_command:.2f} m/s", 
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
        )
        if dones.any():
            pixels = cv2.putText(
                pixels, "FALL DETECTED - RESETTING", 
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3
            )
            
        writer.append_data(pixels)
        
        sys.stdout.write(f"\r  Progress: {t_curr:4.1f}s / {duration}s | Speed: {current_vx_command:.2f} m/s")
        sys.stdout.flush()
        
        if dones.any():
            obs, _ = env.reset()
            
    print("\n✅ Saved.")
    writer.close()
    renderer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=str, help="Path to training logs folder")
    args = parser.parse_args()

    checkpoint_path = find_latest_checkpoint(args.folder_path)
    
    # Save videos directly in Model_Evaluations/<subfolder>/<folder_name>/videos
    run_folder = Path(args.folder_path)
    folder_name = run_folder.name
    subfolder = "With_Torso" if "With_Torso" in args.folder_path else "No_Torso"
    output_dir = Path(__file__).resolve().parents[2] / "Model_Evaluations" / subfolder / folder_name / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎬 Initializing environment for {folder_name}")
    env, policy = setup_env("Mjlab-Velocity-Flat-Angad", checkpoint_path)
    
    cameras = {
        "side": {"distance": 2.0, "azimuth": 90, "elevation": -5},
        "front": {"distance": 2.0, "azimuth": 180, "elevation": -5},
        "isometric": {"distance": 2.5, "azimuth": 135, "elevation": -15},
    }
    
    for name, cam_config in cameras.items():
        out_file = output_dir / f"{name}_view.mp4"
        record_video(env, policy, out_file, cam_config)
        
    print(f"🎉 All videos saved to {output_dir}")

if __name__ == "__main__":
    main()
