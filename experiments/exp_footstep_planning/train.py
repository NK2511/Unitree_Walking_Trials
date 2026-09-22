#!/usr/bin/env python3
"""Train script for Angad footstep planning.

Usage (from this directory):
    /path/to/mjlab_env/bin/python train.py Mjlab-Footstep-Flat-Angad \
        --agent.experiment_name angad_footstep \
        --env.scene.num_envs 4096
"""

# -- Warm-start checkpoint ---------------------------------------------------
# Disabled: the Asimov velocity policy has a different obs ordering
# (base_lin_vel / velocity_commands) than the footstep policy, so partial
# column copy maps the wrong features into the wrong input slots.
# Train from scratch with the corrected reward structure instead.
PRETRAINED_CHECKPOINT: str | None = None

import logging
import os
import sys

# ── Auto-switch to mjlab_env virtual environment ──────────────────────────────
VENV_PYTHON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../mjlab_env/bin/python")
)
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    print("🔄 Switching to mjlab_env Python environment...")
    os.execl(VENV_PYTHON, VENV_PYTHON, *sys.argv)
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

# Add the Unitree_Asimov_Trials/Walk directory to PYTHONPATH so unitree_constants
# and mjlab.tasks imports work correctly.
_WALK_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../Unitree_Asimov_Trials/Walk")
)
if _WALK_DIR not in sys.path:
    sys.path.append(_WALK_DIR)


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlOnPolicyRunnerCfg
    registry_name: str | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    enable_nan_guard: bool = False
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg   = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def _partial_load_pretrained(runner, checkpoint_path: str, device: str) -> None:
    """Transfer weights from a pre-trained walking checkpoint into the runner.

    For layers with exactly matching shapes -> copied as-is.
    For the input weight matrix (mlp.0.weight) where the footstep policy has
    MORE input columns than the pretrained policy (due to extra footstep_targets
    and gait_clock observations) -> the pretrained columns are copied into the
    first N columns and the new columns are set to near-zero so they start
    silent and are learnt gradually.  This preserves the entire base-state
    processing pathway rather than discarding it with a random first layer.
    """
    import torch

    if not os.path.exists(checkpoint_path):
        print(f"[Warm-start]  Checkpoint not found, skipping: {checkpoint_path}")
        return

    print(f"[Warm-start] Loading pretrained weights from:\n  {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Normalise old checkpoint format if needed
    if "model_state_dict" in state and "actor_state_dict" not in state:
        print("[Warm-start] Converting old checkpoint format ...")
        model_sd = state["model_state_dict"]
        actor_sd: dict = {}
        critic_sd: dict = {}
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
        state["actor_state_dict"]  = actor_sd
        state["critic_state_dict"] = critic_sd

    def _copy_matching(current_model, pretrained_sd: dict, label: str) -> None:
        current_sd = current_model.state_dict()
        new_sd = {}
        exact, partial, skipped = 0, 0, 0

        for k in current_sd:
            cur_v = current_sd[k]
            pre_v = pretrained_sd.get(k, None)

            if pre_v is None:
                new_sd[k] = cur_v
                continue

            if cur_v.shape == pre_v.shape:
                # Exact shape match: copy directly
                new_sd[k] = pre_v.to(cur_v.device)
                exact += 1

            elif (cur_v.dim() == 2
                  and pre_v.dim() == 2
                  and cur_v.shape[0] == pre_v.shape[0]
                  and cur_v.shape[1] > pre_v.shape[1]):
                # Input weight extended (more input features in current policy).
                # Copy the pretrained columns; new-obs columns start near-zero
                # so they are effectively silent at the start and learn gradually.
                n_pre = pre_v.shape[1]
                new_w = cur_v.clone()
                new_w[:, :n_pre] = pre_v.to(cur_v.device)
                new_w[:, n_pre:] = new_w[:, n_pre:] * 0.01
                new_sd[k] = new_w
                partial += 1
                n_new = cur_v.shape[1] - n_pre
                print(f"  [Warm-start] Partial {label}/{k}: "
                      f"copied {n_pre} cols, {n_new} new cols near-zero")
            else:
                new_sd[k] = cur_v
                print(f"  [Warm-start] Skip {label}/{k}  "
                      f"pretrained={tuple(pre_v.shape)} vs current={tuple(cur_v.shape)}")
                skipped += 1

        current_model.load_state_dict(new_sd, strict=True)
        print(f"[Warm-start] {label}: {exact} exact, {partial} partial, {skipped} skipped")

    # runner.alg is the PPO instance; _raw_actor/_raw_critic are the uncompiled
    # nn.Module objects (safe for state_dict operations).
    _copy_matching(runner.alg._raw_actor, state.get("actor_state_dict", {}), "actor")
    _copy_matching(runner.alg._raw_critic, state.get("critic_state_dict", {}), "critic")

    print("[Warm-start] Done -- policy warm-started from walking checkpoint.\n")


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        seed   = cfg.agent.seed
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        seed   = cfg.agent.seed + local_rank

    configure_torch_backends()
    cfg.agent.seed = seed
    cfg.env.seed   = seed

    print(f"[INFO] Training with: device={device}, seed={seed}")
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = ManagerBasedRlEnv(
        cfg=cfg.env, device=device,
        render_mode="rgb_array" if cfg.video else None,
    )

    # ── Inject the FootstepManager ────────────────────────────────────────────
    from config.footstep_env_cfg import _patch_env_with_footstep_manager
    plan_path = os.environ.get("FOOTSTEP_PLAN_PATH", None)
    _patch_env_with_footstep_manager(env, plan_path=plan_path)

    # ── Hook the reset callback so FootstepManager is updated on episode end ──
    orig_reset = env.reset

    def _patched_reset(*args, **kwargs):
        obs, info = orig_reset(*args, **kwargs)
        if hasattr(env, "footstep_manager") and env.termination_manager is not None:
            terminated_ids = env.termination_manager.terminated.nonzero(
                as_tuple=False
            ).flatten()
            env.footstep_manager.reset(terminated_ids)
        return obs, info

    env.reset = _patched_reset  # type: ignore[method-assign]

    log_root_path = log_dir.parent

    resume_path: Path | None = None
    if cfg.agent.resume:
        resume_path = get_checkpoint_path(
            log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
        )

    if cfg.video:
        env = VideoRecorder(
            env,
            video_folder=Path(log_dir) / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = asdict(cfg.agent)
    # Map legacy policy config to rsl-rl v3.x format
    if "policy" in agent_cfg and "actor" not in agent_cfg:
        policy_cfg = agent_cfg.pop("policy")
        agent_cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("actor_hidden_dims", (512, 256, 128)),
            "activation":  policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("actor_obs_normalization", False),
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std":  policy_cfg.get("init_noise_std", 1.0),
                "std_type":  policy_cfg.get("noise_std_type", "scalar"),
            },
        }
        agent_cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("critic_hidden_dims", (512, 256, 128)),
            "activation":  policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("critic_obs_normalization", False),
        }
    if "obs_groups" in agent_cfg:
        obs_groups = agent_cfg["obs_groups"]
        if "policy" in obs_groups and "actor" not in obs_groups:
            obs_groups["actor"] = obs_groups.pop("policy")

    # Default to tensorboard logger to avoid wandb 401 unauthorized errors
    if agent_cfg.get("logger") == "wandb" or "logger" not in agent_cfg:
        agent_cfg["logger"] = "tensorboard"

    env_cfg = asdict(cfg.env)

    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, agent_cfg, str(log_dir), device)

    runner.add_git_repo_to_log(__file__)

    # ── Warm-start from pre-trained walking policy ────────────────────────────
    if PRETRAINED_CHECKPOINT is not None and resume_path is None:
        _partial_load_pretrained(runner, PRETRAINED_CHECKPOINT, device)

    if resume_path is not None:
        print(f"[INFO] Loading checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    dump_yaml(log_dir / "params" / "env.yaml",   env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations,
        init_at_random_ep_len=True,
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

    run_train(task_id, args, log_dir)


def main():
    import mjlab.tasks  # noqa: F401
    import config       # noqa: F401

    all_tasks = list_tasks()
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
