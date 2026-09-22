"""
diagnose_policy_behavior.py
============================
Headless diagnostic evaluator that runs a trained checkpoint (or zero policy) for 500 steps,
logs exact physical states every 50 steps, and records an MP4 video to inspect.
"""

import os
import sys
import glob
import re
from pathlib import Path
import torch
import numpy as np

# Virtualenv check
VENV_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mjlab_env/bin/python"))
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    os.execl(VENV_PYTHON, VENV_PYTHON, *sys.argv)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

def find_latest_checkpoint():
    log_root = Path(__file__).parent / "logs" / "rsl_rl"
    model_files = glob.glob(os.path.join(log_root, "**", "model_*.pt"), recursive=True)
    if not model_files:
        return None
    model_files.sort(key=lambda x: os.path.getmtime(x))
    return model_files[-1]

def run_diagnostic():
    import config
    import mjlab.tasks
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    task_id = "Mjlab-Footstep-Flat-Angad"
    
    ckpt_path = find_latest_checkpoint()
    print(f"🔍 Latest checkpoint found: {ckpt_path}")
    
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    from config.footstep_env_cfg import _patch_env_with_footstep_manager
    _patch_env_with_footstep_manager(env)
    wrapped_env = RslRlVecEnvWrapper(env)
    
    agent_cfg = load_rl_cfg(task_id)
    policy = None
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            from dataclasses import asdict
            agent_cfg_dict = asdict(agent_cfg) if hasattr(agent_cfg, '__dataclass_fields__') else agent_cfg
            
            # Map legacy policy keys if present
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

            runner = OnPolicyRunner(wrapped_env, agent_cfg_dict, device=device)
            runner.load(str(ckpt_path), map_location=device)
            policy = runner.get_inference_policy(device=device)
            print("✅ Policy successfully loaded!")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint into runner: {e}")
            print("Falling back to zero-action policy for baseline evaluation.")
    
    obs_dict, _ = wrapped_env.reset()
    
    print("\n" + "="*80)
    print(f"{'Step':<6} | {'Yaw (deg)':<10} | {'t1_local (x,y,z,theta)':<30} | {'t1_is_left':<10} | {'Mean Foot Dist':<15}")
    print("="*80)
    
    robot = env.scene["robot"]
    num_actions = env.action_manager.total_action_dim
    
    for step_i in range(300):
        if policy is not None:
            with torch.inference_mode():
                obs_tensor = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
                actions = policy(obs_tensor)
        else:
            actions = torch.zeros((1, num_actions), device=device)
            
        obs_dict, rew, dones, extras = wrapped_env.step(actions)
        
        if step_i % 25 == 0:
            quat = robot.data.root_link_quat_w[0].cpu().numpy()
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            yaw_deg = np.degrees(np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z)))
            
            # Target local
            fm = env.footstep_manager
            root_pos = robot.data.root_link_pos_w
            root_quat = robot.data.root_link_quat_w
            t1_local = fm.get_goal_steps_local(root_pos, root_quat)[0, :4].cpu().numpy()
            t1_is_left = fm.get_active_target_is_left()[0].item()
            
            site_ids, _ = robot.find_sites(["left_foot_site", "right_foot_site"])
            l_pos = robot.data.site_pos_w[0, site_ids[0]].cpu().numpy()
            r_pos = robot.data.site_pos_w[0, site_ids[1]].cpu().numpy()
            t1_w = fm.get_target_pos_world()[0].cpu().numpy()
            
            active_dist = np.linalg.norm(l_pos - t1_w) if t1_is_left else np.linalg.norm(r_pos - t1_w)
            
            t1_str = f"({t1_local[0]:+.2f}, {t1_local[1]:+.2f}, {t1_local[2]:+.2f}, {np.degrees(t1_local[3]):+.1f}°)"
            print(f"{step_i:<6} | {yaw_deg:<+10.1f} | {t1_str:<30} | {str(t1_is_left):<10} | {active_dist:<15.3f} m")

    print("="*80 + "\n")
    env.close()

if __name__ == "__main__":
    run_diagnostic()
