"""Unified script to train RL agent with RSL-RL and MJLab for Unitree G1."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = (
    cfg.env.commands is not None
    and "motion" in cfg.env.commands
    and isinstance(cfg.env.commands["motion"], MotionCommandCfg)
  )

  if is_tracking_task:
    if not cfg.registry_name:
      raise ValueError("Must provide --registry-name for tracking tasks.")

    # Check if the registry name includes alias, if not, append ":latest".
    registry_name = cast(str, cfg.registry_name)
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)

    assert cfg.env.commands is not None
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", cfg.env)
    dump_yaml(log_dir / "params" / "agent.yaml", cfg.agent)

  env_cfg = cfg.env
  agent_cfg = cfg.agent

  render_mode = "rgb_array" if (cfg.video and rank == 0) else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  # Add video recorder if video flag is set.
  if cfg.video and rank == 0:
    print("[INFO] Recording videos during training.")
    video_kwargs = {
      "video_folder": str(log_dir / "videos" / "train"),
      "step_trigger": lambda step: step % cfg.video_interval == 0,
      "video_length": cfg.video_length,
      "disable_logger": True,
    }
    env = VideoRecorder(env, **video_kwargs)

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  agent_cfg_dict = asdict(agent_cfg)

  # Map policy configuration to rsl-rl-lib format
  if "policy" in agent_cfg_dict and "actor" not in agent_cfg_dict:
    policy_cfg = agent_cfg_dict.pop("policy")
    agent_cfg_dict["actor"] = {
      "class_name": "MLPModel",
      "hidden_dims": policy_cfg.get("actor_hidden_dims", (256, 256, 128)),
      "activation": policy_cfg.get("activation", "elu"),
      "obs_normalization": policy_cfg.get("actor_obs_normalization", False),
      "distribution_cfg": {
        "class_name": "GaussianDistribution",
        "init_std": policy_cfg.get("init_noise_std", 1.0),
        "std_type": policy_cfg.get("noise_std_type", "scalar"),
      },
    }
    agent_cfg_dict["critic"] = {
      "class_name": "MLPModel",
      "hidden_dims": policy_cfg.get("critic_hidden_dims", (256, 256, 128)),
      "activation": policy_cfg.get("activation", "elu"),
      "obs_normalization": policy_cfg.get("critic_obs_normalization", False),
    }
  if "obs_groups" in agent_cfg_dict:
    obs_groups = agent_cfg_dict["obs_groups"]
    if "policy" in obs_groups and "actor" not in obs_groups:
      obs_groups["actor"] = obs_groups.pop("policy")

  runner = runner_cls(env, agent_cfg_dict, log_dir=str(log_dir), device=device)

  resume_path = None
  if agent_cfg.resume:
    if cfg.wandb_run_path is not None:
      resume_path = get_wandb_checkpoint_path(
        cfg.wandb_run_path, agent_cfg.load_run, agent_cfg.load_checkpoint
      )
    else:
      resume_path = get_checkpoint_path(
        log_root_path=log_dir.parent,
        run_dir=agent_cfg.load_run,
        checkpoint=agent_cfg.load_checkpoint,
      )

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  log_root_path = Path(__file__).parent / "logs" / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    run_train(task_id, args, log_dir)
  else:
    import torchrunx
    logging.basicConfig(level=logging.INFO)
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


DEFAULT_TASK_ID = "Mjlab-Velocity-Flat-Unitree"


def main():
  import mjlab.tasks  # noqa: F401
  import envs.unitree_walk  # noqa: F401

  all_tasks = list_tasks()

  # Handle top-level --help or missing task argument gracefully
  if len(sys.argv) < 2 or (sys.argv[1] not in all_tasks and sys.argv[1].startswith("-")):
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
      print(f"Usage: python train.py [TASK_ID] [OPTIONS]")
      print(f"Default TASK_ID: {DEFAULT_TASK_ID}")
      print("\nAvailable Unitree tasks:")
      for t in all_tasks:
        if "Unitree" in t:
          print(f"  - {t}")
      print("\nFor task-specific options, pass the task name followed by --help, e.g.:")
      print(f"  python train.py {DEFAULT_TASK_ID} --help\n")
      return
    # Default to Mjlab-Velocity-Flat-Unitree if flags are passed directly (e.g. --video)
    sys.argv.insert(1, DEFAULT_TASK_ID)
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
