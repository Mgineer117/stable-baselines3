"""Source-faithful intrinsic-reward providers owned by :class:`IRPO`."""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import torch
from gymnasium import spaces
from torch import Tensor, nn

IntrinsicReward = Literal["random", "allo", "lirpg", "drnd"]
PolicyEvaluate = Callable[[dict[str, Tensor], Tensor, Tensor], tuple[Tensor, Tensor | None]]


class _RunningVariance(nn.Module):
    """Variance-only reward normalizer matching the research reward RMS use."""

    def __init__(self, shape: int = 1) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, values: Tensor) -> None:
        values = values.detach().reshape(-1, self.mean.numel()).float()
        if values.numel() == 0:
            return
        batch_count = torch.tensor(float(values.shape[0]), device=values.device)
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        count = self.count.to(values.device)
        delta = batch_mean - self.mean.to(values.device)
        total = count + batch_count
        mean = self.mean.to(values.device) + delta * batch_count / total
        m2 = self.var.to(values.device) * count + batch_var * batch_count + delta.square() * count * batch_count / total
        self.mean.copy_(mean.to(self.mean.device))
        self.var.copy_((m2 / total).to(self.var.device))
        self.count.copy_(total.to(self.count.device))

    def normalize_var_only(self, values: Tensor, update: bool = False) -> Tensor:
        if update:
            self.update(values)
        return values / torch.sqrt(self.var.to(values.device) + 1e-8)


def _is_image(space: spaces.Box) -> bool:
    return len(space.shape) == 3


def _obs(observations: Tensor, image: bool) -> Tensor:
    value = observations.float()
    if image and value.numel() and value.detach().amax() > 1:
        value = value / 255.0
    return value


class _StateEncoder(nn.Module):
    """Nature CNN for images and the research MLP width for vector states."""

    def __init__(self, observation_space: spaces.Box, output_dim: int, depth: int = 1) -> None:
        super().__init__()
        self.image = _is_image(observation_space)
        if self.image:
            channels, height, width = observation_space.shape
            self.body = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten(),
            )
            with torch.no_grad():
                flattened = self.body(torch.zeros(1, channels, height, width)).shape[1]
            self.head = nn.Sequential(nn.Linear(flattened, output_dim), nn.ReLU())
        else:
            input_dim = int(np.prod(observation_space.shape))
            layers: list[nn.Module] = [nn.Flatten()]
            width = 512
            for index in range(depth):
                layers.extend((nn.Linear(input_dim if index == 0 else width, width), nn.ReLU()))
            layers.extend((nn.Linear(width, output_dim), nn.ReLU()))
            self.body = nn.Sequential(*layers)
            self.head = nn.Identity()

    def forward(self, observations: Tensor) -> Tensor:
        return self.head(self.body(_obs(observations, self.image)))


class _ValueNetwork(nn.Module):
    def __init__(self, observation_space: spaces.Box) -> None:
        super().__init__()
        self.encoder = _StateEncoder(observation_space, 512, depth=2)
        self.value = nn.Linear(512, 1)

    def forward(self, observations: Tensor) -> Tensor:
        return self.value(self.encoder(observations))


def _gae(
    rewards: Tensor,
    terminations: Tensor,
    truncations: Tensor,
    values: Tensor,
    next_values: Tensor,
    gamma: float,
    gae_lambda: float,
    n_steps: int,
    n_envs: int,
) -> tuple[Tensor, Tensor]:
    """GAE over time-major VecEnv rollouts, retaining time-limit bootstrap."""

    shape = (n_steps, n_envs)
    reward = rewards.reshape(shape)
    terminated = terminations.reshape(shape)
    truncated = truncations.reshape(shape)
    value = values.reshape(shape)
    next_value = next_values.reshape(shape)
    advantages = torch.zeros_like(reward)
    running = torch.zeros(n_envs, device=rewards.device)
    for index in range(n_steps - 1, -1, -1):
        done = torch.maximum(terminated[index], truncated[index])
        delta = reward[index] + gamma * next_value[index] * (1.0 - terminated[index]) - value[index]
        running = delta + gamma * gae_lambda * running * (1.0 - done)
        advantages[index] = running
    return advantages.flatten(), (value + advantages).flatten()


class IntrinsicRewardProvider(nn.Module):
    """Per-option source-compatible intrinsic reward interface."""

    def rewards(self, batch: Any, options: list[int]) -> dict[int, Tensor]:
        raise NotImplementedError

    def update(
        self,
        batch: Any,
        option: int,
        *,
        policy_evaluate: PolicyEvaluate | None = None,
        params: dict[str, Tensor] | None = None,
        subpolicy_learning_rate: float | None = None,
        external_advantages: Tensor | None = None,
    ) -> dict[str, float]:
        return {}


class RandomReward(IntrinsicRewardProvider):
    """Fixed signed eigen-coordinate potentials with shared reward RMS."""

    def __init__(self, observation_space: spaces.Box, num_options: int) -> None:
        super().__init__()
        self.num_options = num_options
        self.encoder = _StateEncoder(observation_space, max(16, num_options // 2 + 2), depth=4)
        self.reward_rms = _RunningVariance(num_options)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def rewards(self, batch: Any, options: list[int]) -> dict[int, Tensor]:
        with torch.no_grad():
            difference = self.encoder(batch.next_observations) - self.encoder(batch.observations)
            raw = torch.stack([
                difference[:, option // 2 + 1] * (1 if option % 2 else -1)
                for option in range(self.num_options)
            ], dim=1)
            normalized = self.reward_rms.normalize_var_only(raw, update=True)
        return {option: normalized[:, option] for option in options}


class _DRNDModel(nn.Module):
    """The source DRND predictor and ten frozen target networks."""

    def __init__(self, observation_space: spaces.Box, feature_dim: int, targets: int = 10) -> None:
        super().__init__()
        image = _is_image(observation_space)
        if image:
            self.predictor = nn.Sequential(
                _StateEncoder(observation_space, 512),
                nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, feature_dim),
            )
            self.target = nn.ModuleList([_StateEncoder(observation_space, feature_dim) for _ in range(targets)])
        else:
            input_dim = int(np.prod(observation_space.shape))
            self.predictor = nn.Sequential(
                nn.Flatten(), nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, feature_dim),
            )
            self.target = nn.ModuleList([
                nn.Sequential(nn.Flatten(), nn.Linear(input_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, feature_dim))
                for _ in range(targets)
            ])
        self.image = image
        for target in self.target:
            for parameter in target.parameters():
                parameter.requires_grad_(False)

    def forward(self, next_observations: Tensor) -> tuple[Tensor, Tensor]:
        observations = _obs(next_observations, self.image)
        predictor = self.predictor(observations)
        targets = torch.stack([target(observations) for target in self.target], dim=0)
        return predictor, targets


class _DRNDSlot(nn.Module):
    def __init__(self, observation_space: spaces.Box, learning_rate: float, critic_learning_rate: float, feature_dim: int, gamma: float, gae_lambda: float, update_proportion: float) -> None:
        super().__init__()
        self.drnd = _DRNDModel(observation_space, feature_dim)
        self.reward_rms = _RunningVariance()
        self.update_proportion = update_proportion
        self.optimizer = torch.optim.Adam(self.drnd.parameters(), lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)

    def raw_reward(self, next_observations: Tensor) -> Tensor:
        prediction, targets = self.drnd(next_observations)
        mean, second = targets.mean(dim=0), targets.square().mean(dim=0)
        first = 0.9 * (prediction - mean).square().sum(dim=1)
        denominator = (second - mean.square()).abs().clamp_min(1e-6)
        second_term = 0.1 * torch.sqrt(((prediction.square() - mean.square()).abs() / denominator).clamp(1e-6, 1)).sum(dim=1)
        return first + second_term


class DRNDReward(IntrinsicRewardProvider):
    def __init__(
        self,
        observation_space: spaces.Box,
        num_options: int,
        learning_rate: float,
        gamma: float,
        gae_lambda: float,
        feature_dim: int = 16,
        critic_learning_rate: float = 1e-4,
        update_proportion: float = 0.25,
    ) -> None:
        super().__init__()
        self.slots = nn.ModuleList([
            _DRNDSlot(observation_space, learning_rate, critic_learning_rate, feature_dim, gamma, gae_lambda, update_proportion)
            for _ in range(num_options)
        ])

    def rewards(self, batch: Any, options: list[int]) -> dict[int, Tensor]:
        result = {}
        with torch.no_grad():
            for option in options:
                slot = self.slots[option]
                result[option] = slot.reward_rms.normalize_var_only(slot.raw_reward(batch.next_observations), update=False)
        return result

    def update(self, batch: Any, option: int, **_: Any) -> dict[str, float]:
        slot = self.slots[option]
        prediction, targets = slot.drnd(batch.next_observations)
        target_index = torch.randint(targets.shape[0], (batch.next_observations.shape[0],), device=prediction.device)
        target_features = targets[target_index, torch.arange(batch.next_observations.shape[0], device=prediction.device)].detach()
        error = (prediction - target_features).square().mean(dim=-1)
        mask = (torch.rand_like(error) < slot.update_proportion).float()
        drnd_loss = (error * mask).sum() / mask.sum().clamp_min(1)
        with torch.no_grad():
            rewards = slot.reward_rms.normalize_var_only(slot.raw_reward(batch.next_observations), update=False)
        slot.optimizer.zero_grad()
        drnd_loss.backward()
        torch.nn.utils.clip_grad_norm_(slot.drnd.parameters(), 0.5)
        slot.optimizer.step()
        slot.lr_scheduler.step()
        with torch.no_grad():
            slot.reward_rms.update(slot.raw_reward(batch.next_observations))
        return {
            "loss": drnd_loss.item(),
            "predictor_loss": drnd_loss.item(),
            "intrinsic_reward_mean": rewards.mean().item(),
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "providers": [
                {"optimizer": slot.optimizer.state_dict(), "scheduler": slot.lr_scheduler.state_dict()}
                for slot in self.slots
            ]
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        for slot, provider_state in zip(self.slots, state.get("providers", [])):
            slot.optimizer.load_state_dict(provider_state["optimizer"])
            slot.lr_scheduler.load_state_dict(provider_state["scheduler"])


class _LIRPGRewardNetwork(nn.Module):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Space) -> None:
        super().__init__()
        self.discrete = isinstance(action_space, spaces.Discrete)
        self.encoder = _StateEncoder(observation_space, 512, depth=2)
        if self.discrete:
            self.head = nn.Linear(512, action_space.n)
        elif isinstance(action_space, spaces.Box):
            action_dim = int(np.prod(action_space.shape))
            self.head = nn.Sequential(nn.Linear(512 + action_dim, 512), nn.ReLU(), nn.Linear(512, 1))
        else:
            raise NotImplementedError("LIRPG supports Box and Discrete actions")

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        features = self.encoder(states)
        if self.discrete:
            values = self.head(features)
            action = actions.long().flatten()
            return torch.tanh(values.gather(1, action[:, None]).flatten())
        return torch.tanh(self.head(torch.cat((features, actions.float().flatten(1)), dim=1)).flatten())


class _LIRPGSlot(nn.Module):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Space, learning_rate: float, critic_learning_rate: float, optimizer: str) -> None:
        super().__init__()
        self.reward = _LIRPGRewardNetwork(observation_space, action_space)
        optimizer_cls = torch.optim.RMSprop if optimizer == "rmsprop" else torch.optim.Adam
        self.optimizer = optimizer_cls(self.reward.parameters(), lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)


class LIRPGReward(IntrinsicRewardProvider):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        num_options: int,
        learning_rate: float,
        gamma: float,
        gae_lambda: float,
        r_ex_coef: float = 1.0,
        r_in_coef: float = 0.01,
        v_ex_coef: float = 0.5,
        clip_range: float = 0.2,
        critic_learning_rate: float = 1e-4,
        optimizer: str = "adam",
    ) -> dict[str, float]:
        super().__init__()
        self.slots = nn.ModuleList([
            _LIRPGSlot(observation_space, action_space, learning_rate, critic_learning_rate, optimizer)
            for _ in range(num_options)
        ])
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.r_ex_coef, self.r_in_coef, self.v_ex_coef, self.clip_range = r_ex_coef, r_in_coef, v_ex_coef, clip_range

    def rewards(self, batch: Any, options: list[int]) -> dict[int, Tensor]:
        with torch.no_grad():
            return {option: self.slots[option].reward(batch.observations, batch.actions) for option in options}

    def update(
        self,
        batch: Any,
        option: int,
        *,
        policy_evaluate: PolicyEvaluate | None = None,
        params: dict[str, Tensor] | None = None,
        subpolicy_learning_rate: float | None = None,
        external_advantages: Tensor | None = None,
    ) -> None:
        if policy_evaluate is None or params is None or subpolicy_learning_rate is None or external_advantages is None:
            raise ValueError("LIRPG requires final subpolicy parameters and external GAE advantages")
        slot = self.slots[option]
        external_advantage = external_advantages.detach()
        intrinsic_rewards = slot.reward(batch.observations, batch.actions)
        mixed = self.r_ex_coef * external_advantage + self.r_in_coef * intrinsic_rewards
        mixed = (mixed - mixed.mean()) / (mixed.std(unbiased=False) + 1e-8)
        log_prob, _ = policy_evaluate(params, batch.observations, batch.actions)
        ratio = torch.exp(log_prob - batch.log_probs)
        inner_loss = -torch.minimum(ratio * mixed, ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * mixed).mean()
        gradients = torch.autograd.grad(inner_loss, tuple(params.values()), create_graph=True, allow_unused=True)
        lookahead = {
            name: value - subpolicy_learning_rate * (gradient if gradient is not None else torch.zeros_like(value))
            for (name, value), gradient in zip(params.items(), gradients)
        }
        lookahead_log_prob, _ = policy_evaluate(lookahead, batch.observations, batch.actions)
        external_advantage = (external_advantage - external_advantage.mean()) / (external_advantage.std(unbiased=False) + 1e-8)
        lookahead_ratio = torch.exp(lookahead_log_prob - batch.log_probs)
        meta_loss = -torch.minimum(
            lookahead_ratio * external_advantage,
            lookahead_ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * external_advantage,
        ).mean()
        slot.optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(slot.reward.parameters(), 0.5)
        slot.optimizer.step()
        slot.lr_scheduler.step()
        return {
            "loss": meta_loss.item(),
            "meta_loss": meta_loss.item(),
            "intrinsic_reward_mean": intrinsic_rewards.mean().item(),
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "providers": [
                {"optimizer": slot.optimizer.state_dict(), "scheduler": slot.lr_scheduler.state_dict()}
                for slot in self.slots
            ]
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        for slot, provider_state in zip(self.slots, state.get("providers", [])):
            slot.optimizer.load_state_dict(provider_state["optimizer"])
            slot.lr_scheduler.load_state_dict(provider_state["scheduler"])


class ALLOReward(IntrinsicRewardProvider):
    """Pretrained ALLO temporal-eigenfunction encoder, frozen for IRPO."""

    def __init__(
        self,
        observation_space: spaces.Box,
        num_options: int,
        learning_rate: float,
        discount: float = 0.999,
        pretrain_updates: int = 10_000,
        lr_barrier_coeff: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_options = num_options
        self.encoder = _StateEncoder(observation_space, max(10, num_options // 2 + 2), depth=4)
        self.reward_rms = _RunningVariance(num_options)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / pretrain_updates)
        )
        self.discount = discount
        self.lr_barrier_coeff = lr_barrier_coeff
        self.register_buffer("duals", torch.zeros(self.encoder_output_dim, self.encoder_output_dim))
        self.register_buffer("barrier", torch.tril(2 * torch.ones(self.encoder_output_dim, self.encoder_output_dim)))
        self.register_buffer("dual_velocities", torch.zeros(self.encoder_output_dim, self.encoder_output_dim))

    @property
    def encoder_output_dim(self) -> int:
        return max(10, self.num_options // 2 + 2)

    def _sample_pairs(self, observations: Tensor, next_observations: Tensor, dones: Tensor, size: int) -> tuple[Tensor, Tensor]:
        steps, envs = observations.shape[:2]
        starts: list[Tensor] = []
        ends: list[Tensor] = []
        for _ in range(size):
            env = int(torch.randint(envs, ()).item())
            start = int(torch.randint(steps, ()).item())
            available = 1
            while start + available < steps and not dones[start + available - 1, env]:
                available += 1
            if available > 1:
                distances = torch.arange(1, available, device=observations.device)
                distance = int(torch.multinomial(self.discount ** distances, 1).item() + 1)
                starts.append(observations[start, env])
                ends.append(observations[start + distance, env])
            else:
                starts.append(observations[start, env])
                ends.append(next_observations[start, env])
        return torch.stack(starts), torch.stack(ends)

    def pretrain(self, observations: Tensor, next_observations: Tensor, dones: Tensor, updates: int, batch_size: int = 256) -> None:
        self.train()
        for _ in range(updates):
            first, second = self._sample_pairs(observations, next_observations, dones, batch_size)
            phi_first, phi_second = self.encoder(first), self.encoder(second)
            graph_loss = (phi_first - phi_second).square().mean(dim=0).sum()
            flat = observations.flatten(0, 1)
            sample = flat[torch.randint(flat.shape[0], (2 * batch_size,), device=flat.device)]
            independent_first, independent_second = self.encoder(sample[:batch_size]), self.encoder(sample[batch_size:])
            identity = torch.eye(self.encoder_output_dim, device=flat.device)
            error_first = torch.tril(independent_first.T @ independent_first.detach() / batch_size - identity)
            error_second = torch.tril(independent_second.T @ independent_second.detach() / batch_size - identity)
            error = 0.5 * (error_first + error_second)
            loss = graph_loss + (self.duals.detach() * error).sum() + self.barrier[0, 0].detach() * (error_first * error_second).sum()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            with torch.no_grad():
                updated_duals = (self.duals + 1e-3 * torch.tril(error)).clamp(0, 100)
                delta = updated_duals - self.duals
                update_rate = 1.0 if torch.linalg.vector_norm(self.dual_velocities) == 0 else 0.1
                self.dual_velocities.add_(update_rate * (delta - self.dual_velocities))
                self.duals.copy_(torch.tril(updated_duals))
                self.barrier.copy_((self.barrier + self.lr_barrier_coeff * (error_first * error_second).clamp_min(0).mean()).clamp(0, 100))
        self.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def rewards(self, batch: Any, options: list[int]) -> dict[int, Tensor]:
        with torch.no_grad():
            difference = self.encoder(batch.next_observations) - self.encoder(batch.observations)
            raw = torch.stack([
                difference[:, option // 2 + 1] * (1 if option % 2 else -1)
                for option in range(self.num_options)
            ], dim=1)
            normalized = self.reward_rms.normalize_var_only(raw, update=True)
        return {option: normalized[:, option] for option in options}

    def get_extra_state(self) -> dict[str, Any]:
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.lr_scheduler.state_dict()}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            self.lr_scheduler.load_state_dict(state["scheduler"])


def make_intrinsic_reward(
    kind: IntrinsicReward,
    observation_space: spaces.Box,
    action_space: spaces.Space,
    num_options: int,
    *,
    gamma: float,
    gae_lambda: float,
    lirpg_learning_rate: float,
    drnd_learning_rate: float,
    allo_learning_rate: float,
    lirpg_r_ex_coef: float,
    lirpg_r_in_coef: float,
    lirpg_v_ex_coef: float,
    drnd_feature_dim: int,
    allo_pretrain_updates: int,
) -> IntrinsicRewardProvider:
    if kind == "random":
        return RandomReward(observation_space, num_options)
    if kind == "drnd":
        return DRNDReward(observation_space, num_options, drnd_learning_rate, gamma, gae_lambda, drnd_feature_dim)
    if kind == "lirpg":
        return LIRPGReward(
            observation_space, action_space, num_options, lirpg_learning_rate, gamma, gae_lambda,
            lirpg_r_ex_coef, lirpg_r_in_coef, lirpg_v_ex_coef,
        )
    if kind == "allo":
        return ALLOReward(observation_space, num_options, allo_learning_rate, pretrain_updates=allo_pretrain_updates)
    raise ValueError("intrinsic_reward must be 'random', 'allo', 'lirpg', or 'drnd'")
