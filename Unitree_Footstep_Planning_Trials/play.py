"""Script to evaluate and visualize trained Angad footstep planning policies."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import glob
import re

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

# Auto-relaunch inside the correct virtual environment if not already using it
VENV_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mjlab_env/bin/python"))
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    print(f"🔄 Auto-switching to mjlab_env Python interpreter...")
    os.execl(VENV_PYTHON, VENV_PYTHON, *sys.argv)

DEFAULT_TASK_ID = "Mjlab-Footstep-Flat-Angad"


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  num_envs: int | None = 1  # 1 environment by default for visualization
  device: str | None = None
  viewer: Literal["native", "viser"] = "native"
  video: bool = False
  plan_path: str | None = None


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  resume_path = Path(cfg.checkpoint_file) if cfg.checkpoint_file else None
  if resume_path is None:
      # Auto-detect most recent model checkpoint in logs/rsl_rl/angad_footstep/
      log_root = Path(__file__).parent / "logs" / "rsl_rl"
      model_files = glob.glob(os.path.join(log_root, "**", "model_*.pt"), recursive=True)
      
      valid_models = []
      for mf in model_files:
          mtime = os.path.getmtime(mf)
          match = re.search(r"model_(\d+)\.pt", os.path.basename(mf))
          iter_num = int(match.group(1)) if match else 0
          valid_models.append((mtime, iter_num, mf))
      
      if valid_models:
          valid_models.sort(key=lambda x: x[0])  # Sort by modification time
          latest_model_path = valid_models[-1][2]
          latest_iter = valid_models[-1][1]
          resume_path = Path(latest_model_path)
          print(f"🔄 Auto-detected latest trained model: {resume_path} (Iteration {latest_iter})")
      else:
          print(f"⚠️ No model_*.pt files found in {log_root}")

  render_mode = "rgb_array" if cfg.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  from config.footstep_env_cfg import _patch_env_with_footstep_manager
  _patch_env_with_footstep_manager(env, plan_path=cfg.plan_path)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if cfg.agent == "zero":
      def policy(obs):
          return torch.zeros((env.num_envs, env.num_actions), device=device)
  elif cfg.agent == "random":
      def policy(obs):
          return torch.rand((env.num_envs, env.num_actions), device=device) * 2.0 - 1.0
  else:
      runner_cls = load_runner_cls(task_id) or OnPolicyRunner
      agent_cfg_dict = asdict(agent_cfg)
      
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
              
      runner = runner_cls(env, agent_cfg_dict, device=device)
      if resume_path is None:
          raise ValueError("Must provide --checkpoint-file or have a valid saved model in logs/")
      runner.load(str(resume_path), map_location=device)
      policy = runner.get_inference_policy(device=device)

  print("\n=========================================")
  print(" 🎬 LAUNCHING MUJOCO FOOTSTEP VIEWER")
  print(" Active target    : Bright Green sphere")
  print(" Lookahead target : Orange sphere")
  print(" Future targets   : Semi-transparent Blue/Red spheres")
  print("=========================================\n")

  if cfg.viewer == "native":
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()

  env.close()

def main():
  import mjlab.tasks
  import config

  if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].startswith("-")):
      sys.argv.insert(1, DEFAULT_TASK_ID)

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )

  run_play(chosen_task, args)

if __name__ == "__main__":
  main()
