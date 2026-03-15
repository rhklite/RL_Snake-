"""Actor-Critic networks for PPO on the Snake environment."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0
) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class GridActorCritic(nn.Module):
    """CNN-based actor-critic for grid observations (rows × cols int8 grid)."""

    def __init__(self, rows: int, cols: int, n_actions: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(1, 16, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        flat_size = 32 * rows * cols
        self.critic = nn.Sequential(
            layer_init(nn.Linear(flat_size, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(flat_size, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, n_actions), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1).float()
        features = self.encoder(x)
        return self.actor(features), self.critic(features)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1).float()
        return self.critic(self.encoder(x))

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1).float()
        features = self.encoder(x)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(features)


class FeatureActorCritic(nn.Module):
    """MLP-based actor-critic for the 11-d feature observation."""

    def __init__(self, obs_dim: int = 11, n_actions: int = 4) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
        )
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)
        self.actor = layer_init(nn.Linear(128, n_actions), std=0.01)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        return self.actor(features), self.critic(features)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.shared(x))

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(features)


def make_agent(obs_type: str, rows: int = 12, cols: int = 12) -> nn.Module:
    """Factory to create the right network for the given observation type."""
    if obs_type == "grid":
        return GridActorCritic(rows, cols)
    return FeatureActorCritic()
