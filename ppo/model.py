from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_latent_var: int):
        super().__init__()

        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1),
        )

    def forward(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def act(
        self, state: np.ndarray, evaluate: bool = False
    ) -> tuple[int, torch.Tensor]:
        state_t = torch.from_numpy(state).float()
        action_probs = self.action_layer(state_t)
        dist = Categorical(action_probs)

        if evaluate:
            _, action = action_probs.max(0)
        else:
            action = dist.sample()

        return action.item(), dist.log_prob(action)

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy
