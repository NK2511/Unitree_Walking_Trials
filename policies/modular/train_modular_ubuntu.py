"""Training Launcher for Modular Multi-Expert Unitree G1 on Ubuntu.

Run this script on your Ubuntu GPU workstation to train the modular policy.

Usage:
  python train_modular_ubuntu.py --task Mjlab-Velocity-Flat-Unitree --num_envs 4096 --headless
"""

import os
import sys
import argparse
import torch

# Ensure local modules and repository root are accessible
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mjlab.tasks  # noqa: F401
import envs.unitree_walk  # noqa: F401

from modular_architecture import ModularActor, ModularCritic
from rsl_rl_wrapper import RslRlModularActorCritic
from config_modular import ModularRunnerCfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train Modular Unitree G1 Policy on Ubuntu")
    parser.add_argument("--task", type=str, default="Mjlab-Velocity-Flat-Unitree", help="Task ID")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel simulation environments")
    parser.add_argument("--max_iterations", type=int, default=10000, help="Total PPO training iterations")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without graphical viewer")
    parser.add_argument("--log_dir", type=str, default="./logs/modular_unitree", help="Directory to save checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("  UNITREE G1 - MODULAR MULTI-EXPERT REINFORCEMENT LEARNING")
    print("=" * 70)
    print(f"Task:               {args.task}")
    print(f"Parallel Envs:      {args.num_envs}")
    print(f"Compute Device:     {args.device}")
    print(f"Max Iterations:     {args.max_iterations}")
    print(f"Log Directory:      {args.log_dir}")
    print("-" * 70)
    print("Neural Architecture: Approach 1 (Decoupled Sub-Brains)")
    print("  Module 1: Balance & Posture Expert      [IMU + Gravity + Tilt]")
    print("  Module 2: Gait & Stepping Expert        [Joints + Phase Clock]")
    print("  Module 3: Command & Navigation Expert   [Joystick + Smoothing]")
    print("  Arbitration: Dynamic Gating Network     [Live Attention %]")
    print("=" * 70)

    os.makedirs(args.log_dir, exist_ok=True)

    try:
        # Load environment via MJLab or Isaac
        from mjlab.tasks.registry import load_env_cfg
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        print("\nLoading MJLab environment configuration...")
        env_cfg = load_env_cfg(args.task, play=False)
        env_cfg.scene.num_envs = args.num_envs

        print(f"Spawning {args.num_envs} simulated humanoid environments on {args.device}...")
        base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
        env = RslRlVecEnvWrapper(base_env)

        # Prepare Runner Config
        runner_cfg = ModularRunnerCfg()
        runner_cfg.max_iterations = args.max_iterations
        runner_dict = {
            "policy": {
                "class_name": "RslRlModularActorCritic",
                "init_noise_std": 0.5,
                "actor_hidden_dims": (64, 64),
                "critic_hidden_dims": (256, 128),
                "activation": "elu",
            },
            "algorithm": {
                "class_name": "PPO",
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "learning_rate": 1.0e-3,
                "schedule": "adaptive",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
                "value_loss_coef": 1.0,
                "entropy_coef": 0.01,
            },
            "num_steps_per_env": 24,
            "save_interval": 50,
            "experiment_name": "unitree_g1_modular",
            "logger": "tensorboard",
        }

        print("\nInitializing PPO Runner with Modular Actor-Critic...")
        runner = OnPolicyRunner(env=env, train_cfg=runner_dict, log_dir=args.log_dir, device=args.device)

        # Plug in our custom modular policy
        num_obs = env.num_obs
        num_actions = env.num_actions
        print(f"Observation space: {num_obs} dims | Action space: {num_actions} dims")
        
        custom_policy = RslRlModularActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_obs,
            num_actions=num_actions
        ).to(args.device)

        runner.alg.policy = custom_policy

        print("\nStarting Training! Check Tensorboard for real-time loss and reward logs:")
        print(f"  tensorboard --logdir {args.log_dir}\n")
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

        print("\nTraining completed successfully! Policy saved to:", args.log_dir)

    except ImportError as e:
        print("\n[NOTE] Simulation framework import message:", e)
        print("To run on Ubuntu, ensure MJLab / Isaac Gym and RSL-RL are in your python environment:")
        print("  pip install -e ./asimov-mjlab")
        print("  pip install rsl-rl")


if __name__ == "__main__":
    main()
