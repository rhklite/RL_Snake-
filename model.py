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


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


def _get_activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation {name!r}. Choose from: {list(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name]()


class GridActorCritic(nn.Module):
    """CNN-based actor-critic for grid observations (rows x cols int8 grid)."""

    def __init__(
        self,
        rows: int,
        cols: int,
        n_actions: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        channels = [1] + [min(16 * (2**i), 64) for i in range(num_layers)]
        conv_layers: list[nn.Module] = []
        for i in range(num_layers):
            conv_layers.append(
                layer_init(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1)
                )
            )
            conv_layers.append(_get_activation(activation))
        conv_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*conv_layers)

        flat_size = channels[-1] * rows * cols
        self.critic = nn.Sequential(
            layer_init(nn.Linear(flat_size, hidden_size)),
            _get_activation(activation),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(flat_size, hidden_size)),
            _get_activation(activation),
            layer_init(nn.Linear(hidden_size, n_actions), std=0.01),
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

    def __init__(
        self,
        obs_dim: int = 11,
        n_actions: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            layer_init(nn.Linear(obs_dim, hidden_size)),
            _get_activation(activation),
        ]
        for _ in range(num_layers - 1):
            layers.append(layer_init(nn.Linear(hidden_size, hidden_size)))
            layers.append(_get_activation(activation))
        self.shared = nn.Sequential(*layers)
        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)
        self.actor = layer_init(nn.Linear(hidden_size, n_actions), std=0.01)

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


class MLPGridActorCritic(FeatureActorCritic):
    """MLP actor-critic that flattens a grid observation."""

    def __init__(
        self,
        rows: int,
        cols: int,
        n_actions: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__(
            obs_dim=rows * cols,
            n_actions=n_actions,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x.float().flatten(1))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return super().get_value(x.float().flatten(1))

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().get_action_and_value(x.float().flatten(1), action)


class HybridActorCritic(nn.Module):
    """Dual-encoder actor-critic for hybrid observations (CNN grid + MLP food vector).

    Input:
        obs: dict with keys:
            "grid" -- (B, 3, rows, cols) float32 tensor
                ch0: body gradient (tail~0 -> head=1.0),
                ch1: head location (binary),
                ch2: food location (binary).
            "food" -- (B, 2) float32 tensor (normalized dx/cols, dy/rows)
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        n_actions: int = 4,
        hidden_size: int = 128,
        num_layers: int = 4,
        activation: str = "relu",
        feat_hidden: int = 64,
    ) -> None:
        super().__init__()

        channels = [3] + [min(16 * (2**i), 128) for i in range(num_layers)]
        conv_layers: list[nn.Module] = []
        for i in range(num_layers):
            conv_layers.append(
                layer_init(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1)
                )
            )
            conv_layers.append(_get_activation(activation))
            if i < num_layers - 1:
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        conv_layers.append(nn.Flatten())
        self.cnn_encoder = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, rows, cols)
            flat_cnn = self.cnn_encoder(dummy).shape[1]

        self.cnn_proj = nn.Sequential(
            layer_init(nn.Linear(flat_cnn, hidden_size)),
            _get_activation(activation),
        )

        self.feat_encoder = nn.Sequential(
            layer_init(nn.Linear(2, feat_hidden)),
            _get_activation(activation),
        )

        fused_size = hidden_size + feat_hidden
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(fused_size, hidden_size)),
            _get_activation(activation),
        )

        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)
        self.actor = layer_init(nn.Linear(hidden_size, n_actions), std=0.01)

    def _encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        grid = obs["grid"].float()
        food = obs["food"].float()
        cnn_feat = self.cnn_proj(self.cnn_encoder(grid))
        food_feat = self.feat_encoder(food)
        return self.fusion(torch.cat([cnn_feat, food_feat], dim=-1))

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(self._encode(obs))

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._encode(obs)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(features)


def make_agent(
    arch: str,
    obs_type: str,
    rows: int = 12,
    cols: int = 12,
    hidden_size: int = 128,
    num_layers: int = 2,
    activation: str = "relu",
) -> nn.Module:
    """Factory to create the right network for the given config."""
    if arch == "cnn":
        if obs_type != "grid":
            raise ValueError("CNN architecture requires obs_type='grid'")
        return GridActorCritic(
            rows,
            cols,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
        )
    if arch == "mlp":
        if obs_type == "grid":
            return MLPGridActorCritic(
                rows,
                cols,
                hidden_size=hidden_size,
                num_layers=num_layers,
                activation=activation,
            )
        return FeatureActorCritic(
            hidden_size=hidden_size, num_layers=num_layers, activation=activation
        )
    if arch == "hybrid":
        if obs_type != "hybrid":
            raise ValueError("Hybrid architecture requires obs_type='hybrid'")
        return HybridActorCritic(
            rows,
            cols,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
        )
    raise ValueError(f"Unknown architecture: {arch!r}. Use 'cnn', 'mlp', or 'hybrid'.")
