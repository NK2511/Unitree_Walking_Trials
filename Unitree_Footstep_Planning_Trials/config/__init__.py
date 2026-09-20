"""Task registry for Angad footstep planning environment."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .unitree_g1_cfg import unitree_g1_footstep_env_cfg
from .footstep_env_cfg import angad_footstep_flat_env_cfg

# ── RL agent config for Unitree G1 ────────────────────────────────────────────
from dataclasses import dataclass, field

@dataclass
class FootstepRlCfg(RslRlOnPolicyRunnerCfg):
    experiment_name: str = "unitree_g1_footstep"
    run_name: str = ""
    max_iterations: int = 5000
    save_interval: int = 100
    num_steps_per_env: int = 88  # 2 full gait cycles: 2×2.20s / 0.020s per step = 220, use 88 for speed
    policy: RslRlPpoActorCriticCfg = field(default_factory=lambda: RslRlPpoActorCriticCfg(
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
    ))
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=lambda: RslRlPpoAlgorithmCfg(
        learning_rate=1e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01,
        value_loss_coef=1.0,
        max_grad_norm=1.0,
        clip_param=0.2,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
    ))


# ── Task registration ─────────────────────────────────────────────────────────
register_mjlab_task(
    task_id="Mjlab-Footstep-Flat-Angad",
    env_cfg=unitree_g1_footstep_env_cfg(),
    play_env_cfg=unitree_g1_footstep_env_cfg(play=True),
    rl_cfg=FootstepRlCfg(),
)
