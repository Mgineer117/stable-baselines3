"""Intrinsic Reward Policy Optimization for Stable-Baselines3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from gymnasium import spaces
from torch import Tensor, nn
from torch.func import functional_call

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.irpo.intrinsic import IntrinsicReward, LIRPGReward, make_intrinsic_reward


@dataclass
class _Batch:
    observations: Tensor
    next_observations: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    n_steps: int
    n_envs: int


class _PolicyAction(nn.Module):
    def __init__(self, policy: ActorCriticPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.policy(observations)


class _PolicyEvaluation(nn.Module):
    def __init__(self, policy: ActorCriticPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        return self.policy.evaluate_actions(observations, actions)


class IRPO(OnPolicyAlgorithm):
    """Intrinsic Reward Policy Optimization.

    IRPO collects one base rollout and then adapts one functional subpolicy per
    intrinsic-reward option. The external-return objective is differentiated
    through those inner updates and updates the base SB3 policy.

    The initial port uses the paper's differentiable SGD meta update. The TRPO
    natural-gradient update from the research code is intentionally not exposed
    until it can share SB3's policy/distribution machinery without a second
    policy implementation.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 128,
        gamma: float = 0.99,
        ent_coef: float = 0.0,
        intrinsic_reward: IntrinsicReward = "random",
        num_options: int = 3,
        num_subpolicy_updates: int = 5,
        inner_learning_rate: float | None = None,
        intrinsic_learning_rate: float = 7e-4,
        drnd_learning_rate: float = 1e-4,
        allo_encoder_path: str | None = None,
        aggregation_method: Literal["uniform", "softmax", "argmax"] = "softmax",
        temperature: float = 1.0,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: torch.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        if num_options < 1:
            raise ValueError("num_options must be positive")
        if num_subpolicy_updates < 2:
            raise ValueError("num_subpolicy_updates must be at least 2")
        if aggregation_method not in {"uniform", "softmax", "argmax"}:
            raise ValueError("aggregation_method must be 'uniform', 'softmax', or 'argmax'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not isinstance(env, str) and isinstance(env.observation_space, spaces.Dict):
            raise NotImplementedError("The initial IRPO port supports Box observations only")

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=1.0,
            ent_coef=ent_coef,
            vf_coef=0.0,
            max_grad_norm=0.0,
            use_sde=False,
            sde_sample_freq=-1,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Box, spaces.Discrete),
        )
        self.num_options = num_options
        self.num_subpolicy_updates = num_subpolicy_updates
        self.inner_learning_rate = inner_learning_rate
        self.intrinsic_learning_rate = intrinsic_learning_rate
        self.aggregation_method = aggregation_method
        self.temperature = temperature
        self.intrinsic_reward_kind = intrinsic_reward
        self.allo_encoder_path = allo_encoder_path
        self.drnd_learning_rate = drnd_learning_rate
        self._lirpg_optimizer: torch.optim.Optimizer | None = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self.intrinsic_provider = make_intrinsic_reward(
            self.intrinsic_reward_kind,
            self.num_options,
            allo_encoder_path=self.allo_encoder_path,
            drnd_learning_rate=self.drnd_learning_rate,
        ).to(self.device)
        self._action_module = _PolicyAction(self.policy)
        self._evaluation_module = _PolicyEvaluation(self.policy)

    def _params(self) -> dict[str, Tensor]:
        return {f"policy.{name}": parameter for name, parameter in self.policy.named_parameters()}

    def _action(self, params: dict[str, Tensor], observation: np.ndarray) -> np.ndarray:
        tensor = obs_as_tensor(observation, self.device)
        with torch.no_grad():
            actions, _, _ = functional_call(self._action_module, params, (tensor,))
        return actions.cpu().numpy()

    def _collect_batch(
        self,
        env: VecEnv,
        params: dict[str, Tensor],
        callback: Any | None,
    ) -> _Batch | None:
        observation = env.reset()
        observations: list[np.ndarray] = []
        next_observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        dones: list[np.ndarray] = []

        for _ in range(self.n_steps):
            sampled_actions = self._action(params, observation)
            env_actions = sampled_actions
            if isinstance(self.action_space, spaces.Box):
                env_actions = np.clip(sampled_actions, self.action_space.low, self.action_space.high)
            new_observation, reward, done, infos = env.step(env_actions)
            successor = np.array(new_observation, copy=True)
            for index, is_done in enumerate(done):
                terminal = infos[index].get("terminal_observation") if is_done else None
                if terminal is not None:
                    successor[index] = terminal

            observations.append(np.array(observation, copy=True))
            next_observations.append(successor)
            actions.append(np.array(sampled_actions, copy=True))
            rewards.append(np.asarray(reward, dtype=np.float32).copy())
            dones.append(np.asarray(done, dtype=np.float32).copy())
            observation = new_observation
            self.num_timesteps += env.num_envs
            if callback is not None:
                callback.update_locals(locals())
                if not callback.on_step():
                    return None

        def tensor(values: list[np.ndarray]) -> Tensor:
            return torch.as_tensor(np.asarray(values), device=self.device)

        obs = tensor(observations)
        next_obs = tensor(next_observations)
        action = tensor(actions)
        if isinstance(self.action_space, spaces.Discrete):
            action = action.long()
        return _Batch(
            observations=obs.flatten(0, 1),
            next_observations=next_obs.flatten(0, 1),
            actions=action.flatten(0, 1),
            rewards=tensor(rewards).flatten(0, 1),
            dones=tensor(dones).flatten(0, 1),
            n_steps=self.n_steps,
            n_envs=env.num_envs,
        )

    def _returns(self, rewards: Tensor, dones: Tensor, n_steps: int, n_envs: int) -> Tensor:
        shaped = rewards.reshape(n_steps, n_envs)
        ended = dones.reshape(n_steps, n_envs)
        returns = torch.empty_like(shaped)
        running = torch.zeros(n_envs, device=self.device)
        for index in range(n_steps - 1, -1, -1):
            running = shaped[index] + self.gamma * running * (1.0 - ended[index])
            returns[index] = running
        flat = returns.flatten()
        return (flat - flat.mean()) / (flat.std(unbiased=False) + 1e-8)

    def _policy_loss(self, params: dict[str, Tensor], batch: _Batch, advantages: Tensor) -> Tensor:
        actions = batch.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()
        _, log_prob, entropy = functional_call(
            self._evaluation_module, params, (batch.observations, actions)
        )
        loss = -(log_prob * advantages).mean()
        if entropy is not None:
            loss -= self.ent_coef * entropy.mean()
        return loss

    def _adapt(self, params: dict[str, Tensor], batch: _Batch, option: int, final: bool) -> dict[str, Tensor]:
        rewards = batch.rewards if final else self.intrinsic_provider.reward(
            batch.observations, batch.next_observations, option
        )
        advantages = self._returns(rewards, batch.dones, batch.n_steps, batch.n_envs)
        loss = self._policy_loss(params, batch, advantages)
        values = tuple(params.values())
        gradients = torch.autograd.grad(loss, values, create_graph=True, allow_unused=True)
        learning_rate = self.inner_learning_rate or self.lr_schedule(self._current_progress_remaining)
        return {
            name: value - learning_rate * (gradient if gradient is not None else torch.zeros_like(value))
            for (name, value), gradient in zip(params.items(), gradients)
        }

    def _weights(self, scores: Tensor) -> Tensor:
        if self.aggregation_method == "uniform":
            return torch.full_like(scores, 1.0 / len(scores))
        if self.aggregation_method == "argmax":
            weights = torch.zeros_like(scores)
            weights[scores.argmax()] = 1.0
            return weights
        return torch.softmax(scores / self.temperature, dim=0)

    def _update_lirpg(self, loss: Tensor) -> None:
        if not isinstance(self.intrinsic_provider, LIRPGReward):
            return
        parameters = tuple(self.intrinsic_provider.parameters())
        if self._lirpg_optimizer is None:
            self._lirpg_optimizer = torch.optim.RMSprop(parameters, lr=self.intrinsic_learning_rate)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        self._lirpg_optimizer.zero_grad()
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = None if gradient is None else gradient.detach()
        self._lirpg_optimizer.step()

    def _meta_update(self, loss: Tensor) -> float:
        parameters = tuple(self.policy.parameters())
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        learning_rate = self.lr_schedule(self._current_progress_remaining)
        norm = torch.zeros((), device=self.device)
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                if gradient is not None:
                    parameter.add_(gradient, alpha=-learning_rate)
                    norm += gradient.square().sum()
        return norm.sqrt().item()

    def train(self) -> None:
        """IRPO trains inside :meth:`learn`; required by OnPolicyAlgorithm."""

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "IRPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "IRPO":
        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )
        callback.on_training_start(locals(), globals())
        assert self.env is not None
        iteration = 0

        while self.num_timesteps < total_timesteps:
            callback.on_rollout_start()
            base_params = self._params()
            base_batch = self._collect_batch(self.env, base_params, callback)
            if base_batch is None:
                break
            option_losses: list[Tensor] = []
            scores: list[Tensor] = []
            complete = True
            for option in range(self.num_options):
                params = base_params
                batch = base_batch
                final_batch: _Batch | None = None
                for update in range(self.num_subpolicy_updates):
                    final = update == self.num_subpolicy_updates - 1
                    if update:
                        batch = self._collect_batch(self.env, params, callback)
                        if batch is None:
                            complete = False
                            break
                    params = self._adapt(params, batch, option, final)
                    if final:
                        final_batch = batch
                if not complete or final_batch is None:
                    break
                external_advantage = self._returns(
                    final_batch.rewards, final_batch.dones, final_batch.n_steps, final_batch.n_envs
                )
                option_losses.append(self._policy_loss(params, final_batch, external_advantage))
                scores.append(final_batch.rewards.mean().detach())
                self.intrinsic_provider.update(final_batch.observations, final_batch.next_observations)
            callback.on_rollout_end()
            if not complete:
                break

            score_tensor = torch.stack(scores)
            weights = self._weights(score_tensor)
            meta_loss = torch.sum(torch.stack(option_losses) * weights)
            self._update_lirpg(meta_loss)
            gradient_norm = self._meta_update(meta_loss)
            iteration += 1
            self._n_updates += 1
            self.logger.record("train/meta_loss", meta_loss.item())
            self.logger.record("train/meta_gradient_norm", gradient_norm)
            self.logger.record("train/option_return", score_tensor.mean().item())
            self.logger.record("train/option_weight_max", weights.max().item())
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            if log_interval and iteration % log_interval == 0:
                self.dump_logs(iteration)

        callback.on_training_end()
        return self

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        return ["policy", "intrinsic_provider"], []
