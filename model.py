"""Actor-Critic networks for PPO on the Snake environment."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from snake_env import build_cycle_phase


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
            "grid" -- (B, 4, rows, cols) float32 tensor
                ch0: body gradient (tail~0 -> head=1.0),
                ch1: head location (binary),
                ch2: food location (binary),
                ch3: BFS reachability (inverted distance, 0=unreachable).
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
        adaptive_pool_size: int | None = None,
    ) -> None:
        super().__init__()

        channels = [4] + [min(16 * (2**i), 128) for i in range(num_layers)]
        conv_layers: list[nn.Module] = []
        for i in range(num_layers):
            conv_layers.append(
                layer_init(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1)
                )
            )
            conv_layers.append(_get_activation(activation))
            if adaptive_pool_size is None and i < num_layers - 1:
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if adaptive_pool_size is not None:
            with torch.no_grad():
                pre = nn.Sequential(*conv_layers)(torch.zeros(1, 4, rows, cols))
            if min(pre.shape[-2], pre.shape[-1]) < adaptive_pool_size:
                raise ValueError(
                    f"adaptive_pool_size={adaptive_pool_size} exceeds pre-pool "
                    f"spatial {tuple(pre.shape[-2:])} for {rows}x{cols}; "
                    f"AdaptiveMaxPool would upsample (degenerate)."
                )
            conv_layers.append(nn.AdaptiveMaxPool2d(adaptive_pool_size))
        conv_layers.append(nn.Flatten())
        self.cnn_encoder = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 4, rows, cols)
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
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._encode(obs)
        logits = self.actor(features)
        if action_mask is not None:
            # Illegal actions -> -inf logit (prob 0). Must be passed identically in
            # rollout AND the PPO recompute or the importance ratio is wrong.
            logits = logits.masked_fill(~action_mask, float("-inf"))
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(features)


def _build_grid_encoder(
    in_ch: int,
    rows: int,
    cols: int,
    num_layers: int,
    activation: str,
    adaptive_pool_size: int | None,
) -> tuple[nn.Sequential, int]:
    """Build a hybrid-style CNN grid encoder for ``in_ch`` input channels.

    Mirrors ``HybridActorCritic``'s grid stack but with a configurable input-channel
    count (the asymmetric critic uses 5: the 4 hybrid channels + 1 cycle-phase channel).
    Returns the encoder and its flattened feature dim. Kept separate so the original
    ``HybridActorCritic`` construction (and its checkpoint keys) stay untouched.
    """
    channels = [in_ch] + [min(16 * (2**i), 128) for i in range(num_layers)]
    layers: list[nn.Module] = []
    for i in range(num_layers):
        layers.append(
            layer_init(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1))
        )
        layers.append(_get_activation(activation))
        if adaptive_pool_size is None and i < num_layers - 1:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    if adaptive_pool_size is not None:
        with torch.no_grad():
            pre = nn.Sequential(*layers)(torch.zeros(1, in_ch, rows, cols))
        if min(pre.shape[-2], pre.shape[-1]) < adaptive_pool_size:
            raise ValueError(
                f"adaptive_pool_size={adaptive_pool_size} exceeds pre-pool spatial "
                f"{tuple(pre.shape[-2:])} for {rows}x{cols}; AdaptiveMaxPool would upsample."
            )
        layers.append(nn.AdaptiveMaxPool2d(adaptive_pool_size))
    layers.append(nn.Flatten())
    encoder = nn.Sequential(*layers)
    with torch.no_grad():
        flat = encoder(torch.zeros(1, in_ch, rows, cols)).shape[1]
    return encoder, flat


class AsymmetricHybridActorCritic(nn.Module):
    """Asymmetric actor-critic: privileged critic, plain actor.

    The **actor** is a standard ``HybridActorCritic`` (4-channel hybrid obs) — its
    weights warm-start tensor-for-tensor from a shared-trunk checkpoint, and the
    policy is deployable with the unchanged observation (the cycle is never an actor
    input). The **critic** has its own encoder over the 4 hybrid channels PLUS a
    static, grid-only Hamiltonian cycle-phase channel, so the value function sees the
    canonical traversal order while the actor does not. The cycle field is held as a
    buffer (computed from grid size), so the env obs and the PPO loop are unchanged.
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
        adaptive_pool_size: int | None = None,
    ) -> None:
        super().__init__()
        # Actor pathway: a full HybridActorCritic (we use only its actor head). Same
        # module/key layout as the saved checkpoints -> strict actor warm-start.
        self.actor_net = HybridActorCritic(
            rows,
            cols,
            n_actions=n_actions,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            feat_hidden=feat_hidden,
            adaptive_pool_size=adaptive_pool_size,
        )
        # Critic pathway: separate encoder over grid(4) + cycle-phase(1) and food.
        self.critic_cnn, flat_cnn = _build_grid_encoder(
            5, rows, cols, num_layers, activation, adaptive_pool_size
        )
        self.critic_proj = nn.Sequential(
            layer_init(nn.Linear(flat_cnn, hidden_size)),
            _get_activation(activation),
        )
        self.critic_feat = nn.Sequential(
            layer_init(nn.Linear(2, feat_hidden)),
            _get_activation(activation),
        )
        self.critic_fusion = nn.Sequential(
            layer_init(nn.Linear(hidden_size + feat_hidden, hidden_size)),
            _get_activation(activation),
        )
        self.critic_head = layer_init(nn.Linear(hidden_size, 1), std=1.0)
        # Static privileged channel: normalized Hamiltonian cycle phase per cell.
        phase = build_cycle_phase(rows, cols)  # (rows, cols) float32
        self.register_buffer("cycle_field", torch.from_numpy(phase).unsqueeze(0))

    def _critic_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        grid = obs["grid"].float()  # (B, 4, H, W)
        b, _, h, w = grid.shape
        cyc = self.cycle_field.to(grid.device).expand(b, 1, h, w)
        grid5 = torch.cat([grid, cyc], dim=1)  # (B, 5, H, W)
        cnn_feat = self.critic_proj(self.critic_cnn(grid5))
        food_feat = self.critic_feat(obs["food"].float())
        return self.critic_head(self.critic_fusion(torch.cat([cnn_feat, food_feat], dim=-1)))

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._critic_value(obs)

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Actor sees only the 4-channel hybrid obs (never the cycle field).
        features = self.actor_net._encode(obs)
        logits = self.actor_net.actor(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self._critic_value(obs)

    def load_actor_weights(self, state_dict: dict) -> None:
        """Strict-load actor weights from a HybridActorCritic checkpoint (critic stays fresh)."""
        self.actor_net.load_state_dict(state_dict)


def make_agent(
    arch: str,
    obs_type: str,
    rows: int = 12,
    cols: int = 12,
    hidden_size: int = 128,
    num_layers: int = 2,
    activation: str = "relu",
    adaptive_pool_size: int | None = None,
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
            adaptive_pool_size=adaptive_pool_size,
        )
    if arch == "hybrid_asym":
        if obs_type != "hybrid":
            raise ValueError("hybrid_asym architecture requires obs_type='hybrid'")
        return AsymmetricHybridActorCritic(
            rows,
            cols,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            adaptive_pool_size=adaptive_pool_size,
        )
    raise ValueError(
        f"Unknown architecture: {arch!r}. Use 'cnn', 'mlp', 'hybrid', or 'hybrid_asym'."
    )
