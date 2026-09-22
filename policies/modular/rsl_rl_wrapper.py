"""RSL-RL Integration Wrapper for Modular Actor-Critic.

This module wraps the ModularActor and ModularCritic into an interface
that is 100% plug-and-play compatible with the RSL-RL PPO algorithm
and OnPolicyRunner on Ubuntu.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.distributions import Normal

from modular_architecture import ModularActor, ModularCritic


class RslRlModularActorCritic(nn.Module):
    """Drop-in ActorCritic class for RSL-RL PPO algorithm."""

    is_recurrent: bool = False

    def __init__(
        self,
        num_actor_obs: int = 49,
        num_critic_obs: int = 49,
        num_actions: int = 12,
        actor_hidden_dims: Tuple[int, ...] = (64, 64),
        critic_hidden_dims: Tuple[int, ...] = (256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

        # Modular Actor with 3 decoupled physical experts
        self.actor = ModularActor(
            obs_dim=num_actor_obs,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            balance_hidden=actor_hidden_dims,
            gait_hidden=(128, 64),
            cmd_hidden=actor_hidden_dims,
        )

        # Decomposed Multi-Head Critic
        self.critic = ModularCritic(
            obs_dim=num_critic_obs,
            hidden_dims=critic_hidden_dims,
        )

        # State tracking for inference telemetry
        self.last_gating_weights: Optional[torch.Tensor] = None
        self.last_diagnostics: Optional[Dict[str, torch.Tensor]] = None

    @property
    def action_mean(self) -> torch.Tensor:
        return self._last_action_mean

    @property
    def action_std(self) -> torch.Tensor:
        return torch.exp(self.actor.log_std).expand_as(self._last_action_mean)

    @property
    def entropy(self) -> torch.Tensor:
        std = torch.exp(self.actor.log_std).expand_as(self._last_action_mean)
        return Normal(self._last_action_mean, std).entropy().sum(dim=-1)

    def reset(self, dones: Optional[torch.Tensor] = None):
        """No recurrent hidden states to reset in standard feedforward modular policy."""
        pass

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample action during training with exploration noise."""
        mean_action, weights, diags = self.actor.forward_experts(observations)
        self._last_action_mean = mean_action
        self.last_gating_weights = weights.detach()
        self.last_diagnostics = {k: v.detach() for k, v in diags.items()}

        std = torch.exp(self.actor.log_std).expand_as(mean_action)
        dist = Normal(mean_action, std)
        actions = dist.rsample()
        return actions

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic action forward pass for zero-noise deployment / evaluation."""
        mean_action, weights, diags = self.actor.forward_experts(observations)
        self._last_action_mean = mean_action
        self.last_gating_weights = weights.detach()
        self.last_diagnostics = {k: v.detach() for k, v in diags.items()}
        return mean_action

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluate state value for Generalized Advantage Estimation (GAE)."""
        return self.critic(critic_observations)

    def evaluate_decomposed(self, critic_observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return decomposed reward stream values for physical analysis."""
        return self.critic.evaluate_decomposed(critic_observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate log probability of actions for PPO surrogate loss."""
        std = torch.exp(self.actor.log_std).expand_as(self._last_action_mean)
        dist = Normal(self._last_action_mean, std)
        return dist.log_prob(actions).sum(dim=-1)

    def get_expert_distribution(self) -> Optional[torch.Tensor]:
        """Return the most recent expert attention weights [alpha_bal, alpha_gait, alpha_cmd]."""
        return self.last_gating_weights
