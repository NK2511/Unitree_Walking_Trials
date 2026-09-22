"""Abstract base interface for all actor-critic policies.

Every policy in this PhD project — MLP, Modular, RNN, CPG+Residual —
must subclass BasePolicyInterface so experiments can hot-swap them.

To add a new architecture:
1. Create policies/your_new_policy/your_policy.py
2. Subclass BasePolicyInterface
3. Implement act(), act_inference(), evaluate()
4. Change one import line in your experiment config
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class BasePolicyInterface(ABC, nn.Module):
    """Minimal interface contract for all actor-critic policies.

    Implementing this interface guarantees the policy is compatible
    with the RSL-RL OnPolicyRunner training loop and the project's
    evaluation scripts (policy_evaluator.py, joystick.py).
    """

    is_recurrent: bool = False  # Override to True for RNN policies

    @abstractmethod
    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample an action during training (with exploration noise).

        Args:
            observations: (num_envs, obs_dim) observation tensor

        Returns:
            actions: (num_envs, num_actions) sampled actions
        """

    @abstractmethod
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Return deterministic action for zero-noise deployment / eval.

        Args:
            observations: (num_envs, obs_dim) observation tensor

        Returns:
            actions: (num_envs, num_actions) deterministic actions
        """

    @abstractmethod
    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluate state value for Generalized Advantage Estimation.

        Args:
            critic_observations: (num_envs, critic_obs_dim) tensor

        Returns:
            values: (num_envs, 1) estimated state values
        """

    @abstractmethod
    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log-probability of actions under the current policy.

        Called by PPO to compute the surrogate clipped objective.

        Args:
            actions: (batch_size, num_actions) tensor

        Returns:
            log_probs: (batch_size,) log-probability of each action
        """

    def reset(self, dones: Optional[torch.Tensor] = None):
        """Reset any recurrent hidden states.

        Only needs implementation for RNN-based policies.
        Default: no-op for feedforward policies.
        """
        pass

    def get_interpretability_info(self) -> Optional[dict]:
        """Return any diagnostic / interpretability data from the last forward pass.

        For example, the modular policy returns gating weights here.
        Default: None (black-box MLP has nothing to report).
        """
        return None
