"""Script to play RL agent with RSL-RL and Keyboard Control."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import threading
import time

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

# Auto-relaunch inside the correct virtual environment if not already using it
VENV_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mjlab_env/bin/python"))
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    print(f"🔄 Auto-switching to mjlab_env Python interpreter...")
    os.execl(VENV_PYTHON, VENV_PYTHON, *sys.argv)

# Configurable defaults when running without arguments
DEFAULT_TASK_ID = "Mjlab-Velocity-Flat-Angad"
DEFAULT_CHECKPOINT_FILE = None

# Keyboard control setup
from pynput import keyboard

GLOBAL_VEL_CMD = [0.0, 0.0, 0.0]  # Initial state: stopped
MAX_VX_FORWARD = 4.5
MAX_VX_BACKWARD = 0.8
MAX_VY = 0.6
MAX_WZ = 0.6

LAST_KEY_TIME = 0.0
KEY_COOLDOWN = 0.1  # 100ms cooldown to prevent double keypress triggers

def adjust_velocity(index, delta, limit_min, limit_max):
    GLOBAL_VEL_CMD[index] = round(max(limit_min, min(limit_max, GLOBAL_VEL_CMD[index] + delta)), 2)

def on_press(key):
    global GLOBAL_VEL_CMD, LAST_KEY_TIME
    curr_time = time.time()
    if curr_time - LAST_KEY_TIME < KEY_COOLDOWN:
        return
        
    try:
        if key.char == 'x':
            GLOBAL_VEL_CMD = [0.0, 0.0, 0.0]
            LAST_KEY_TIME = curr_time
            print(f"\n⏹️ Stopped: Cmd [Vx: 0.0, Vy: 0.0, Wz: 0.0]")
    except AttributeError:
        if key == keyboard.Key.up:
            adjust_velocity(0, 0.2, -MAX_VX_BACKWARD, MAX_VX_FORWARD)
            LAST_KEY_TIME = curr_time
        elif key == keyboard.Key.down:
            adjust_velocity(0, -0.2, -MAX_VX_BACKWARD, MAX_VX_FORWARD)
            LAST_KEY_TIME = curr_time
        elif key == keyboard.Key.left:
            adjust_velocity(1, 0.2, -MAX_VY, MAX_VY)
            LAST_KEY_TIME = curr_time
        elif key == keyboard.Key.right:
            adjust_velocity(1, -0.2, -MAX_VY, MAX_VY)
            LAST_KEY_TIME = curr_time
        elif key == keyboard.Key.page_up:
            adjust_velocity(2, 0.2, -MAX_WZ, MAX_WZ)
            LAST_KEY_TIME = curr_time
        elif key == keyboard.Key.page_down:
            adjust_velocity(2, -0.2, -MAX_WZ, MAX_WZ)
            LAST_KEY_TIME = curr_time
        elif key in [keyboard.Key.space, keyboard.Key.backspace]:
            GLOBAL_VEL_CMD = [0.0, 0.0, 0.0]
            LAST_KEY_TIME = curr_time
            print(f"\n⏹️ Stopped: Cmd [Vx: 0.0, Vy: 0.0, Wz: 0.0]")


def on_release(key):
    # Stateful mode: do not reset velocities to 0 on key release
    pass


def keyboard_listener_thread():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

# Start background keyboard thread
threading.Thread(target=keyboard_listener_thread, daemon=True).start()

def custom_key_callback(key: int) -> None:
    global GLOBAL_VEL_CMD, LAST_KEY_TIME
    curr_time = time.time()
    if curr_time - LAST_KEY_TIME < KEY_COOLDOWN:
        return
        
    if key == 265:  # KEY_UP
        adjust_velocity(0, 0.2, -MAX_VX_BACKWARD, MAX_VX_FORWARD)
        LAST_KEY_TIME = curr_time
    elif key == 264:  # KEY_DOWN
        adjust_velocity(0, -0.2, -MAX_VX_BACKWARD, MAX_VX_FORWARD)
        LAST_KEY_TIME = curr_time
    elif key == 263:  # KEY_LEFT
        adjust_velocity(1, 0.2, -MAX_VY, MAX_VY)
        LAST_KEY_TIME = curr_time
    elif key == 262:  # KEY_RIGHT
        adjust_velocity(1, -0.2, -MAX_VY, MAX_VY)
        LAST_KEY_TIME = curr_time
    elif key == 266:  # KEY_PAGE_UP
        adjust_velocity(2, 0.2, -MAX_WZ, MAX_WZ)
        LAST_KEY_TIME = curr_time
    elif key == 267:  # KEY_PAGE_DOWN
        adjust_velocity(2, -0.2, -MAX_WZ, MAX_WZ)
        LAST_KEY_TIME = curr_time
    elif key == 88:  # KEY_X
        GLOBAL_VEL_CMD = [0.0, 0.0, 0.0]
        LAST_KEY_TIME = curr_time


# --- Play Config ---
@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  checkpoint_file: str | None = DEFAULT_CHECKPOINT_FILE
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "native"
  meshes: bool = False
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  # Use the mesh-visualized XML file for play/evaluation if requested
  mesh_xml = Path(__file__).resolve().parents[2] / "xmls" / "unitree_g1" / "scene.xml"
  if cfg.meshes and mesh_xml.exists() and "robot" in env_cfg.scene.entities:
      import unitree_constants
      import mujoco
      def get_mesh_spec() -> mujoco.MjSpec:
          spec = mujoco.MjSpec.from_file(str(mesh_xml))
          return spec
      env_cfg.scene.entities["robot"].spec_fn = get_mesh_spec

  resume_path = Path(cfg.checkpoint_file) if cfg.checkpoint_file else None
  if resume_path is None:
      # Auto-detect most recent model checkpoint
      import glob
      import re
      import unitree_constants
      
      log_root = Path(__file__).parent / "logs" / "rsl_rl" / "unitree_velocity"
      
      # Match run directories starting with YYYY-MM-DD or digits_digits
      run_dirs = sorted([
          d for d in glob.glob(os.path.join(log_root, "*"))
          if os.path.isdir(d) and (re.match(r"^\d{4}-\d{2}-\d{2}", os.path.basename(d)) or re.match(r"^\d+_\d+", os.path.basename(d)))
      ])
      
      if run_dirs:
          latest_run_dir = run_dirs[-1]
          model_files = glob.glob(os.path.join(latest_run_dir, "model_*.pt"))
          
          # Filter out files and extract model iteration numbers
          valid_models = []
          for mf in model_files:
              match = re.search(r"model_(\d+)\.pt", os.path.basename(mf))
              if match:
                  valid_models.append((int(match.group(1)), mf))
          
          if valid_models:
              # Sort by iteration number and get the latest
              valid_models.sort(key=lambda x: x[0])
              latest_model_path = valid_models[-1][1]
              resume_path = Path(latest_model_path)
              print(f"🔄 Auto-detected most recent model in {subfolder}: {resume_path.relative_to(Path(__file__).parent)}")
          else:
              print(f"⚠️ No model_*.pt files found in the latest run directory: {latest_run_dir}")
      else:
          print(f"⚠️ No run directories found in {log_root}")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if cfg.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  
  # Monkey patch the command manager to use our keyboard commands
  from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
  cmd_term = env.unwrapped.command_manager.get_term("twist")
  if isinstance(cmd_term, UniformVelocityCommand):
      print("[INFO] Monkey-patching UniformVelocityCommand for Keyboard Control...")
      print("=========================================")
      print(" KEYBOARD CONTROLS ACTIVE (Native Window Focus):")
      print(" ARROW KEYS        : Forward / Backward / Left / Right")
      print(" PAGE UP / DOWN    : Turn CCW / Turn CW")
      print(" X                 : Stop moving (resets to 0 without resetting sim)")
      print("=========================================")
      
      def custom_resample_command(self, env_ids: torch.Tensor) -> None:
          pass  # Do nothing, we don't want random resampling!

      def custom_update_command(self) -> None:
          global GLOBAL_VEL_CMD
          # Directly set the velocity command from our global variables
          self.vel_command_b[:, 0] = GLOBAL_VEL_CMD[0]
          self.vel_command_b[:, 1] = GLOBAL_VEL_CMD[1]
          self.vel_command_b[:, 2] = GLOBAL_VEL_CMD[2]
          
          # Periodically print the actual vs commanded velocity to the terminal
          if not hasattr(self, "_last_print_time"):
              self._last_print_time = 0.0
              
          curr_time = time.time()
          if curr_time - self._last_print_time > 0.2:  # Update every 200ms
              act_vx = self.robot.data.root_link_lin_vel_b[0, 0].item()
              act_vy = self.robot.data.root_link_lin_vel_b[0, 1].item()
              act_wz = self.robot.data.root_link_ang_vel_b[0, 2].item()
              sys.stdout.write(f"\r🎯 Cmd [X: {GLOBAL_VEL_CMD[0]:+4.1f}  Y: {GLOBAL_VEL_CMD[1]:+4.1f}  Yaw: {GLOBAL_VEL_CMD[2]:+4.1f}]  |  🏃 Act [X: {act_vx:+4.1f}  Y: {act_vy:+4.1f}  Yaw: {act_wz:+4.1f}]   ")
              sys.stdout.flush()
              self._last_print_time = curr_time
      
      # Bind the patched methods
      import types
      cmd_term._resample_command = types.MethodType(custom_resample_command, cmd_term)
      cmd_term._update_command = types.MethodType(custom_update_command, cmd_term)

  if cfg.agent == "zero":
      def policy(obs):
          return torch.zeros((env.num_envs, env.num_actions), device=device)
  elif cfg.agent == "random":
      def policy(obs):
          return torch.rand((env.num_envs, env.num_actions), device=device) * 2.0 - 1.0
  else:
      runner_cls = load_runner_cls(task_id) or OnPolicyRunner
      agent_cfg_dict = asdict(agent_cfg)
      
      # Map legacy policy configuration to rsl-rl-lib v3.x format
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
          raise ValueError("Must provide --checkpoint-file when agent='trained'")
      runner.load(str(resume_path), map_location=device)
      policy = runner.get_inference_policy(device=device)

  resolved_viewer = cfg.viewer
  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy, key_callback=custom_key_callback).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()

  env.close()

def main():
  import mjlab.tasks  # noqa: F401
  import config

  # If run without arguments (e.g. VS Code Play button), inject default task
  if len(sys.argv) < 2:
      sys.argv.append(DEFAULT_TASK_ID)

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
