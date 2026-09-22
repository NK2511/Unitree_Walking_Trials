"""Standard MLP Actor-Critic Baseline Policy for Unitree G1.

This is Experiment 001 — the current trained model.
Architecture: 3-layer MLP (256, 256, 128) with ELU activations.

This wrapper makes the RSL-RL MLPModel conform to BasePolicyInterface
so it can be hot-swapped with modular or RNN policies.
"""

from typing import Optional
import torch
import torch.nn as nn
from torch.distributions import Normal

from policies.base_policy import BasePolicyInterface


class MlpActor(nn.Module):
    """Standard 3-layer ELU MLP Actor. Matches current trained checkpoint."""

    def __init__(self, obs_dim: int, num_actions: int,
                 hidden_dims=(256, 256, 128), activation=nn.ELU,
                 init_noise_std: float = 0.5):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), activation()])
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, num_actions)
        self.log_std = nn.Parameter(torch.full((num_actions,), torch.log(torch.tensor(init_noise_std))))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.output(self.net(obs))


class MlpCritic(nn.Module):
    """Standard 3-layer ELU MLP Critic."""

    def __init__(self, obs_dim: int, hidden_dims=(256, 256, 128), activation=nn.ELU):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), activation()])
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.net(obs))


class MlpPolicy(BasePolicyInterface):
    """Standard monolithic MLP Actor-Critic — Experiment 001 Baseline."""

    is_recurrent: bool = False

    def __init__(self, num_actor_obs: int = 49, num_critic_obs: int = 49,
                 num_actions: int = 12, actor_hidden_dims=(256, 256, 128),
                 critic_hidden_dims=(256, 256, 128), init_noise_std: float = 0.5,
                 **kwargs):
        super().__init__()
        self.actor = MlpActor(num_actor_obs, num_actions, actor_hidden_dims,
                              init_noise_std=init_noise_std)
        self.critic = MlpCritic(num_critic_obs, critic_hidden_dims)
        self._last_mean: Optional[torch.Tensor] = None

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        mean = self.actor(observations)
        self._last_mean = mean
        std = torch.exp(self.actor.log_std).expand_as(mean)
        return Normal(mean, std).rsample()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        mean = self.actor(observations)
        self._last_mean = mean
        return mean

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self._last_mean is not None
        std = torch.exp(self.actor.log_std).expand_as(self._last_mean)
        return Normal(self._last_mean, std).log_prob(actions).sum(dim=-1)

    def get_interpretability_info(self):
        return None  # Black box — nothing to report
