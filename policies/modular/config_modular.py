"""Configuration for Modular RL Training of Unitree G1 on Ubuntu.

This config customizes:
1. Modular policy class registration.
2. Network layer dimensions tailored for the 3 functional experts.
3. PPO algorithm hyperparameters for stable modular learning.
"""

from dataclasses import dataclass, field
from typing import Tuple, Literal


@dataclass
class ModularPolicyCfg:
    """Configuration for the 3-expert modular policy."""
    class_name: str = "RslRlModularActorCritic"
    init_noise_std: float = 0.5
    actor_hidden_dims: Tuple[int, ...] = (64, 64)
    critic_hidden_dims: Tuple[int, ...] = (256, 128)
    activation: str = "elu"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False


@dataclass
class ModularAlgorithmCfg:
    """Hyperparameters for PPO training of modular policy."""
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: Literal["adaptive", "fixed"] = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@dataclass
class ModularRunnerCfg:
    """Runner setup for Ubuntu GPU training."""
    experiment_name: str = "unitree_g1_modular_rl"
    logger: str = "tensorboard"
    save_interval: int = 50
    num_steps_per_env: int = 24
    max_iterations: int = 15_000
    policy: ModularPolicyCfg = field(default_factory=ModularPolicyCfg)
    algorithm: ModularAlgorithmCfg = field(default_factory=ModularAlgorithmCfg)
