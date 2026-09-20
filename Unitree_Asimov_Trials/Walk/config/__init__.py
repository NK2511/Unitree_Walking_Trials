"""Angad velocity task configurations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_rough_env_cfg, unitree_flat_env_cfg
from .rl_cfg import asimov_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Unitree",
    env_cfg=unitree_rough_env_cfg(),
    play_env_cfg=unitree_rough_env_cfg(play=True),
    rl_cfg=asimov_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Unitree",
    env_cfg=unitree_flat_env_cfg(),
    play_env_cfg=unitree_flat_env_cfg(play=True),
    rl_cfg=asimov_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# Backward-compatibility aliases
register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Angad",
    env_cfg=unitree_rough_env_cfg(),
    play_env_cfg=unitree_rough_env_cfg(play=True),
    rl_cfg=asimov_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Angad",
    env_cfg=unitree_flat_env_cfg(),
    play_env_cfg=unitree_flat_env_cfg(play=True),
    rl_cfg=asimov_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
