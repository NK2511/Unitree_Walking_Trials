"""
record_policy_video.py
======================
Renders an MP4 video of the trained policy executing footsteps in MuJoCo,
saves it to the artifacts directory, and logs physical diagnostics.
"""

import os
import sys
import glob
from pathlib import Path
import torch
import numpy as np
import imageio

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

def record_video(max_steps: int = 300, output_path: str = "policy_rollout.mp4"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    task_id = "Mjlab-Footstep-Flat-Angad"

    log_root = Path(__file__).parent / "logs" / "rsl_rl"
    model_files = glob.glob(os.path.join(log_root, "**", "model_*.pt"), recursive=True)
    if not model_files:
        print("❌ No model_*.pt files found!")
        return
    model_files.sort(key=lambda x: os.path.getmtime(x))
    latest_ckpt = model_files[-1]
    print(f"🎬 Loading checkpoint: {latest_ckpt}")

    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    from config.footstep_env_cfg import _patch_env_with_footstep_manager
    _patch_env_with_footstep_manager(env)
    wrapped_env = RslRlVecEnvWrapper(env)

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

    try:
        runner = OnPolicyRunner(wrapped_env, agent_cfg_dict, device=device)
        runner.load(str(latest_ckpt), map_location=device)
        policy = runner.get_inference_policy(device=device)
        print("✅ Policy successfully loaded!")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        return

    obs_dict, _ = wrapped_env.reset()
    frames = []

    print("\n🎥 Rendering video frames...")
    for step_i in range(max_steps):
        with torch.inference_mode():
            obs_tensor = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            actions = policy(obs_tensor)
        
        obs_dict, rew, dones, extras = wrapped_env.step(actions)
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    if frames:
        imageio.mimsave(output_path, frames, fps=30)
        print(f"🎥 Video saved to {output_path} ({len(frames)} frames)")
    else:
        print("⚠️ No frames were rendered.")

    env.close()

if __name__ == "__main__":
    record_video()
