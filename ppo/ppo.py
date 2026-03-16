from __future__ import annotations

import torch
import torch.nn as nn

from ppo.model import ActorCritic
from ppo.memory import Memory


class PPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_latent_var: int,
        lr: float,
        betas: tuple[float, float],
        gamma: float,
        K_epochs: int,
        eps_clip: float,
        entropy_coef: float = 0.01,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var)
        self.policy_old.share_memory()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory: Memory) -> float:
        old_states = memory.states.detach()
        old_actions = memory.actions.detach()
        old_logprobs = memory.logprobs.detach()
        old_disReturn = memory.disReturn.detach()

        std = old_disReturn.std()
        if std == 0:
            old_disReturn = (old_disReturn - old_disReturn.mean()) / 1e-5
        else:
            old_disReturn = (old_disReturn - old_disReturn.mean()) / (std + 1e-8)

        total_loss = 0.0
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = old_disReturn - state_values.detach()

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, old_disReturn)
                - self.entropy_coef * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            total_loss += loss.mean().item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        return total_loss / self.K_epochs
