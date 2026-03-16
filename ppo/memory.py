from __future__ import annotations

import torch
import torch.nn as nn


class Memory:
    """Pre-allocated shared-memory buffer for parallel agent experience collection."""

    def __init__(
        self,
        num_agents: int,
        update_timestep: int,
        state_dim: int,
        agent_policy: nn.Module,
    ):
        total = update_timestep * num_agents

        self.states = torch.zeros((total, state_dim)).share_memory_()
        self.actions = torch.zeros(total).share_memory_()
        self.logprobs = torch.zeros(total).share_memory_()
        self.disReturn = torch.zeros(total).share_memory_()

        self.agent_policy = agent_policy
