"""Intrinsic-reward providers used only by :class:`stable_baselines3.IRPO`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

IntrinsicReward = Literal["random", "allo", "lirpg", "drnd"]


class _RewardNetwork(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(256), nn.ReLU(), nn.Linear(256, output_dim)
        )

    def forward(self, observations: Tensor) -> Tensor:
        x = observations.float()
        if x.numel() and x.detach().amax() > 1:
            x = x / 255.0
        return self.net(x)


class IntrinsicRewardProvider(nn.Module):
    """One stateful intrinsic-reward provider per IRPO option."""

    def reward(self, states: Tensor, next_states: Tensor, option: int) -> Tensor:
        raise NotImplementedError

    def update(self, states: Tensor, next_states: Tensor) -> None:
        """Update after an option rollout. Fixed providers leave this empty."""


class RandomReward(IntrinsicRewardProvider):
    """A fixed random potential function; each option uses one output column."""

    def __init__(self, num_options: int) -> None:
        super().__init__()
        self.features = _RewardNetwork(num_options)
        self._frozen = False

    def reward(self, states: Tensor, next_states: Tensor, option: int) -> Tensor:
        with torch.no_grad():
            next_values = self.features(next_states)
            state_values = self.features(states)
        if not self._frozen:
            for parameter in self.features.parameters():
                parameter.requires_grad_(False)
            self._frozen = True
        return next_values[:, option] - state_values[:, option]


class DRNDReward(IntrinsicRewardProvider):
    """Random-network-distillation prediction error, one target per option."""

    def __init__(self, num_options: int, learning_rate: float = 1e-4) -> None:
        super().__init__()
        self.target = _RewardNetwork(num_options)
        self.predictor = _RewardNetwork(num_options)
        self.learning_rate = learning_rate
        self.optimizer: torch.optim.Optimizer | None = None
        self._target_frozen = False

    def _ensure_optimizer(self) -> None:
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.learning_rate)

    def _error(self, next_states: Tensor) -> Tensor:
        with torch.no_grad():
            target = self.target(next_states)
        if not self._target_frozen:
            for parameter in self.target.parameters():
                parameter.requires_grad_(False)
            self._target_frozen = True
        prediction = self.predictor(next_states)
        return (prediction - target).square()

    def reward(self, states: Tensor, next_states: Tensor, option: int) -> Tensor:
        with torch.no_grad():
            return self._error(next_states)[:, option]

    def update(self, states: Tensor, next_states: Tensor) -> None:
        self._ensure_optimizer()
        assert self.optimizer is not None
        loss = self._error(next_states).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class LIRPGReward(IntrinsicRewardProvider):
    """A learned reward network differentiated through IRPO's inner updates."""

    def __init__(self, num_options: int) -> None:
        super().__init__()
        self.reward_net = _RewardNetwork(num_options)

    def reward(self, states: Tensor, next_states: Tensor, option: int) -> Tensor:
        return torch.tanh(self.reward_net(next_states)[:, option])


class ALLOReward(IntrinsicRewardProvider):
    """Frozen ALLO potential encoder loaded from explicit pretraining output.

    The checkpoint contains an ``encoder`` state dict for ``_RewardNetwork``.
    Each option uses one learned potential dimension and receives its temporal
    difference as intrinsic reward.
    """

    def __init__(self, num_options: int, encoder_path: str) -> None:
        super().__init__()
        path = Path(encoder_path)
        if not path.is_file():
            raise ValueError(f"ALLO checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "encoder" not in checkpoint:
            raise ValueError("ALLO checkpoint must contain an 'encoder' state dict")
        self.encoder = _RewardNetwork(num_options)
        self.encoder.load_state_dict(checkpoint["encoder"])
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def reward(self, states: Tensor, next_states: Tensor, option: int) -> Tensor:
        with torch.no_grad():
            return self.encoder(next_states)[:, option] - self.encoder(states)[:, option]


def make_intrinsic_reward(
    kind: IntrinsicReward,
    num_options: int,
    *,
    allo_encoder_path: str | None = None,
    drnd_learning_rate: float = 1e-4,
) -> IntrinsicRewardProvider:
    if kind == "random":
        return RandomReward(num_options)
    if kind == "drnd":
        return DRNDReward(num_options, drnd_learning_rate)
    if kind == "lirpg":
        return LIRPGReward(num_options)
    if kind == "allo":
        if not allo_encoder_path:
            raise ValueError("ALLO requires allo_encoder_path from explicit pretraining")
        return ALLOReward(num_options, allo_encoder_path)
    raise ValueError("intrinsic_reward must be 'random', 'allo', 'lirpg', or 'drnd'")
