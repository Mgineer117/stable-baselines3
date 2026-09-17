"""Intrinsic Reward Policy Optimization for Stable-Baselines3."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces
from torch import Tensor, nn
from torch.func import functional_call

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.irpo.intrinsic import ALLOReward, IntrinsicReward, _ValueNetwork, _gae, make_intrinsic_reward


@dataclass
class _Batch:
    observations: Tensor
    next_observations: Tensor
    actions: Tensor
    log_probs: Tensor
    rewards: Tensor
    dones: Tensor
    terminations: Tensor
    truncations: Tensor
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


class _PolicyDistribution(nn.Module):
    def __init__(self, policy: ActorCriticPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observations: Tensor) -> torch.distributions.Distribution:
        return self.policy.get_distribution(observations).distribution


class _IRPOHumanOutputFormat(HumanOutputFormat):
    """Render IRPO interval comparisons without changing scalar outputs."""

    def write(self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int = 0) -> None:
        display_values = key_values.get("_irpo_display")
        if isinstance(display_values, dict):
            key_values = {key: display_values.get(key, value) for key, value in key_values.items() if key != "_irpo_display"}
            key_excluded = {key: excluded for key, excluded in key_excluded.items() if key != "_irpo_display"}
        key_values = {f"r~{key}" if key.startswith("int_module/") else key: value for key, value in key_values.items()}
        key_excluded = {f"r~{key}" if key.startswith("int_module/") else key: excluded for key, excluded in key_excluded.items()}
        original_file = self.file
        self.file = _IRPOBoxFile(original_file)
        try:
            super().write(key_values, key_excluded, step)
        finally:
            self.file = original_file


class _IRPOBoxFile:
    """Keep SB3's table rows but make its dash borders explicit."""

    def __init__(self, file: Any) -> None:
        self.file = file

    def write(self, text: str) -> int:
        def pad_ansi(line: str) -> str:
            ansi_width = sum(len(code) for code in re.findall(r"\x1b\[[0-9;]*m", line))
            return f"{line[:-2]}{' ' * ansi_width}{line[-2:]}" if ansi_width and line.endswith(" |") else line

        text = "\n".join(pad_ansi(line) for line in text.split("\n"))
        text = text.replace("r~int_module/", "int_module/  ")
        return self.file.write(re.sub(r"(?m)^-(.+)-$", r"+\1+", text))

    def flush(self) -> None:
        self.file.flush()


class _OptionCriticBank(nn.Module):
    """ASDF-style per-option intrinsic and extrinsic critics."""

    def __init__(
        self, observation_space: spaces.Box, num_options: int, learning_rate: float,
        gamma: float, gae_lambda: float, batch_size: int, n_epochs: int,
    ) -> None:
        super().__init__()
        self.intrinsic = nn.ModuleList(_ValueNetwork(observation_space) for _ in range(num_options))
        self.extrinsic = nn.ModuleList(_ValueNetwork(observation_space) for _ in range(num_options))
        self.intrinsic_optimizers = [torch.optim.Adam(critic.parameters(), lr=learning_rate) for critic in self.intrinsic]
        self.extrinsic_optimizers = [torch.optim.Adam(critic.parameters(), lr=learning_rate) for critic in self.extrinsic]
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.batch_size, self.n_epochs = batch_size, n_epochs

    def targets(self, batch: _Batch, option: int, rewards: Tensor, intrinsic: bool) -> tuple[Tensor, Tensor]:
        critic = (self.intrinsic if intrinsic else self.extrinsic)[option]
        with torch.no_grad():
            advantages, returns = _gae(
                rewards, batch.terminations, batch.truncations, critic(batch.observations).flatten(),
                critic(batch.next_observations).flatten(), self.gamma, self.gae_lambda, batch.n_steps, batch.n_envs,
            )
        return advantages, returns

    def advantages(self, batch: _Batch, option: int, rewards: Tensor, intrinsic: bool, normalize: bool = True) -> Tensor:
        advantages, _ = self.targets(batch, option, rewards, intrinsic)
        return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8) if normalize else advantages

    def update(self, batch: _Batch, option: int, rewards: Tensor, intrinsic: bool, returns: Tensor | None = None) -> float:
        critics = self.intrinsic if intrinsic else self.extrinsic
        optimizers = self.intrinsic_optimizers if intrinsic else self.extrinsic_optimizers
        critic = critics[option]
        if returns is None:
            _, returns = self.targets(batch, option, rewards, intrinsic)
        losses: list[float] = []
        # Match PPO's value fitting: fixed GAE targets, shuffled minibatches,
        # and complete passes over the rollout for each epoch.
        for _ in range(self.n_epochs):
            for indices in torch.randperm(returns.numel(), device=returns.device).split(self.batch_size):
                loss = nn.functional.mse_loss(critic(batch.observations[indices]).flatten(), returns[indices])
                optimizers[option].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                optimizers[option].step()
                losses.append(loss.item())
        return float(np.mean(losses))

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "intrinsic_optimizers": [optimizer.state_dict() for optimizer in self.intrinsic_optimizers],
            "extrinsic_optimizers": [optimizer.state_dict() for optimizer in self.extrinsic_optimizers],
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        for optimizer, value in zip(self.intrinsic_optimizers, state.get("intrinsic_optimizers", [])):
            optimizer.load_state_dict(value)
        for optimizer, value in zip(self.extrinsic_optimizers, state.get("extrinsic_optimizers", [])):
            optimizer.load_state_dict(value)


class IRPO(OnPolicyAlgorithm):
    """SB3-native IRPO with source-compatible outer updates and providers.

    The intentionally retained SB3 simplification is serial option collection.
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
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.98,
        ent_coef: float = 0.0,
        # Subpolicy hyperparameters.
        num_options: int = 2,
        subpolicy_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        critic_batch_size: int = 256,
        critic_n_epochs: int = 5,
        num_subpolicy_updates: int = 3,
        # Intrinsic-reward hyperparameters.
        intrinsic_reward: IntrinsicReward = "lirpg",
        drnd_learning_rate: float = 1e-4,
        allo_learning_rate: float = 1e-4,
        lirpg_learning_rate: float = 5e-5,
        lirpg_r_ex_coef: float = 1.0,
        lirpg_r_in_coef: float = 0.01,
        lirpg_v_ex_coef: float = 0.5,
        drnd_feature_dim: int = 16,
        allo_pretrain_timesteps: int | None = None,
        allo_pretrain_updates: int = 10_000,
        allo_pretrain_collect_batch_size: int = 10_000,
        # Meta-policy hyperparameters.
        temperature: float = 1.0,
        temperature_anneal_timing: float = 0.5,
        performance_ema_beta: float = 0.99,
        target_kl: float = 1e-4,
        trpo_damping: float = 0.1,
        trpo_cg_steps: int = 5,
        trpo_backtrack_iters: int = 10,
        trpo_backtrack_coeff: float = 0.7,
        trpo_batch_size: int = 256,
        # Logging and device.
        evaluation_env: GymEnv | None = None,
        n_eval_episodes: int = 10,
        stats_window_size: int = 100,
        tensorboard_log: str | None = "tensorboard",
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: torch.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        if num_options < 1 or num_subpolicy_updates < 2:
            raise ValueError("num_options must be positive and num_subpolicy_updates must be at least 2")
        if critic_learning_rate <= 0 or critic_batch_size < 1 or critic_n_epochs < 1:
            raise ValueError("critic_learning_rate, critic_batch_size, and critic_n_epochs must be positive")
        if temperature <= 0 or not 0.0 <= temperature_anneal_timing <= 1.0:
            raise ValueError("temperature must be positive and temperature_anneal_timing must be in [0, 1]")
        if not 0.0 <= performance_ema_beta <= 1.0:
            raise ValueError("performance_ema_beta must be in [0, 1]")
        if target_kl <= 0 or trpo_damping < 0 or trpo_cg_steps < 1 or trpo_backtrack_iters < 1 or trpo_batch_size < 1:
            raise ValueError("invalid TRPO hyperparameters")
        if allo_pretrain_updates < 1 or allo_pretrain_collect_batch_size < 1:
            raise ValueError("ALLO pretraining counts must be positive")
        if n_eval_episodes < 1:
            raise ValueError("n_eval_episodes must be positive")
        if not isinstance(env, str) and not isinstance(env.observation_space, spaces.Box):
            raise NotImplementedError("IRPO currently supports Box observations")

        # IRPO experiments always retain TensorBoard scalars; callers may redirect,
        # but may not accidentally disable the only persistent metric stream.
        tensorboard_log = tensorboard_log or "tensorboard"
        super().__init__(
            policy,
            env,
            # SB3 creates a policy optimizer, but IRPO updates the meta-policy with TRPO.
            learning_rate=0.0,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
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
        self.subpolicy_learning_rate = subpolicy_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.critic_batch_size = critic_batch_size
        self.critic_n_epochs = critic_n_epochs
        self.lirpg_learning_rate = lirpg_learning_rate
        self.drnd_learning_rate = drnd_learning_rate
        self.allo_learning_rate = allo_learning_rate
        self.lirpg_r_ex_coef = lirpg_r_ex_coef
        self.lirpg_r_in_coef = lirpg_r_in_coef
        self.lirpg_v_ex_coef = lirpg_v_ex_coef
        self.drnd_feature_dim = drnd_feature_dim
        self.allo_pretrain_timesteps = allo_pretrain_timesteps
        self.allo_pretrain_updates = allo_pretrain_updates
        self.allo_pretrain_collect_batch_size = allo_pretrain_collect_batch_size
        self.allo_pretrained = False
        self._allo_pretraining_steps = 0
        self.temperature = temperature
        self.temperature_anneal_timing = temperature_anneal_timing
        self.performance_ema_beta = performance_ema_beta
        self.performance_gains = torch.zeros(num_options, device=self.device)
        self._active_temperature_anneal_timestep: float | None = None
        self.target_kl = target_kl
        self.trpo_damping = trpo_damping
        self.trpo_cg_steps = trpo_cg_steps
        self.trpo_backtrack_iters = trpo_backtrack_iters
        self.trpo_backtrack_coeff = trpo_backtrack_coeff
        self.trpo_batch_size = trpo_batch_size
        self.intrinsic_reward_kind = intrinsic_reward
        self.evaluation_policy: ActorCriticPolicy | None = None
        if evaluation_env is not None and not isinstance(evaluation_env, VecEnv):
            evaluation_env = DummyVecEnv([lambda: evaluation_env])
        self.evaluation_env = evaluation_env
        self.n_eval_episodes = n_eval_episodes
        self._last_log_iteration = 0
        self._last_logged_values: dict[str, float] = {}

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        if not isinstance(self.observation_space, spaces.Box):
            raise NotImplementedError("IRPO currently supports Box observations")
        self.intrinsic_provider = make_intrinsic_reward(
            self.intrinsic_reward_kind,
            self.observation_space,
            self.action_space,
            self.num_options,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            lirpg_learning_rate=self.lirpg_learning_rate,
            drnd_learning_rate=self.drnd_learning_rate,
            allo_learning_rate=self.allo_learning_rate,
            lirpg_r_ex_coef=self.lirpg_r_ex_coef,
            lirpg_r_in_coef=self.lirpg_r_in_coef,
            lirpg_v_ex_coef=self.lirpg_v_ex_coef,
            drnd_feature_dim=self.drnd_feature_dim,
            allo_pretrain_updates=self.allo_pretrain_updates,
        ).to(self.device)
        self.option_critics = _OptionCriticBank(
            self.observation_space, self.num_options, self.critic_learning_rate, self.gamma, self.gae_lambda,
            self.critic_batch_size, self.critic_n_epochs,
        ).to(self.device)
        self._action_module = _PolicyAction(self.policy)
        self._evaluation_module = _PolicyEvaluation(self.policy)
        self._distribution_module = _PolicyDistribution(self.policy)
        if isinstance(self.intrinsic_provider, ALLOReward):
            if self.allo_pretrained:
                for parameter in self.intrinsic_provider.encoder.parameters():
                    parameter.requires_grad_(False)
            else:
                self._pretrain_allo(self.intrinsic_provider)
                self.allo_pretrained = True
        self.evaluation_policy = deepcopy(self.policy)

    def _pretrain_allo(self, provider: ALLOReward) -> None:
        assert self.env is not None and isinstance(self.observation_space, spaces.Box)
        target = self.allo_pretrain_timesteps
        if target is None:
            target = 100_000 if len(self.observation_space.shape) == 3 else 200_000
        remaining, completed = target, 0
        observation = self.env.reset()
        while remaining > 0:
            transition_count = min(remaining, self.allo_pretrain_collect_batch_size)
            rollout_steps = int(np.ceil(transition_count / self.env.num_envs))
            observations: list[np.ndarray] = []
            successors: list[np.ndarray] = []
            dones: list[np.ndarray] = []
            for _ in range(rollout_steps):
                actions = np.asarray([self.action_space.sample() for _ in range(self.env.num_envs)])
                next_observation, _, done, infos = self.env.step(actions)
                successor = np.array(next_observation, copy=True)
                for index, is_done in enumerate(done):
                    if is_done and infos[index].get("terminal_observation") is not None:
                        successor[index] = infos[index]["terminal_observation"]
                observations.append(np.array(observation, copy=True))
                successors.append(successor)
                dones.append(np.asarray(done, dtype=np.bool_))
                observation = next_observation
            allocated = max(1, round(self.allo_pretrain_updates * transition_count / target))
            provider.pretrain(
                torch.as_tensor(np.asarray(observations), device=self.device),
                torch.as_tensor(np.asarray(successors), device=self.device),
                torch.as_tensor(np.asarray(dones), device=self.device),
                allocated,
            )
            completed += transition_count
            remaining -= transition_count
        self._allo_pretraining_steps = completed

    def _params(self) -> dict[str, Tensor]:
        return {f"policy.{name}": parameter for name, parameter in self.policy.named_parameters()}

    def _action(self, params: dict[str, Tensor], observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = obs_as_tensor(observation, self.device)
        with torch.no_grad():
            actions, _, log_probs = functional_call(self._action_module, params, (tensor,))
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def _collect_batch(self, env: VecEnv, params: dict[str, Tensor], callback: Any | None) -> _Batch | None:
        assert self._last_obs is not None
        observation = self._last_obs
        observations: list[np.ndarray] = []
        next_observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        log_probs: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        terminations: list[np.ndarray] = []
        truncations: list[np.ndarray] = []

        for _ in range(self.n_steps):
            sampled_actions, sampled_log_probs = self._action(params, observation)
            env_actions = np.clip(sampled_actions, self.action_space.low, self.action_space.high) if isinstance(self.action_space, spaces.Box) else sampled_actions
            new_observation, reward, done, infos = env.step(env_actions)
            self._update_info_buffer(infos, done)
            successor = np.array(new_observation, copy=True)
            truncated = np.asarray([bool(info.get("TimeLimit.truncated", False)) and is_done for info, is_done in zip(infos, done)])
            terminated = np.asarray(done, dtype=bool) & ~truncated
            for index, is_done in enumerate(done):
                terminal = infos[index].get("terminal_observation") if is_done else None
                if terminal is not None:
                    successor[index] = terminal
            observations.append(np.array(observation, copy=True))
            next_observations.append(successor)
            actions.append(np.array(sampled_actions, copy=True))
            log_probs.append(np.asarray(sampled_log_probs, dtype=np.float32).copy())
            rewards.append(np.asarray(reward, dtype=np.float32).copy())
            dones.append(np.asarray(done, dtype=np.float32).copy())
            terminations.append(terminated.astype(np.float32))
            truncations.append(truncated.astype(np.float32))
            observation = new_observation
            self._last_obs = observation
            self.num_timesteps += env.num_envs
            if callback is not None:
                callback.update_locals(locals())
                if not callback.on_step():
                    return None

        def tensor(values: list[np.ndarray]) -> Tensor:
            return torch.as_tensor(np.asarray(values), device=self.device)

        action = tensor(actions)
        if isinstance(self.action_space, spaces.Discrete):
            action = action.long()
        return _Batch(
            observations=tensor(observations).flatten(0, 1),
            next_observations=tensor(next_observations).flatten(0, 1),
            actions=action.flatten(0, 1),
            log_probs=tensor(log_probs).flatten(0, 1),
            rewards=tensor(rewards).flatten(0, 1),
            dones=tensor(dones).flatten(0, 1),
            terminations=tensor(terminations).flatten(0, 1),
            truncations=tensor(truncations).flatten(0, 1),
            n_steps=self.n_steps,
            n_envs=env.num_envs,
        )

    def _evaluate_actions(self, params: dict[str, Tensor], observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor | None]:
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()
        _, log_prob, entropy = functional_call(self._evaluation_module, params, (observations, actions))
        return log_prob, entropy

    def _policy_loss(self, params: dict[str, Tensor], batch: _Batch, advantages: Tensor) -> Tensor:
        log_prob, entropy = self._evaluate_actions(params, batch.observations, batch.actions)
        loss = -(log_prob * advantages).mean()
        if entropy is not None:
            loss -= self.ent_coef * entropy.mean()
        return loss

    def _adapt(self, params: dict[str, Tensor], batch: _Batch, advantages: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        loss = self._policy_loss(params, batch, advantages)
        gradients = torch.autograd.grad(loss, tuple(params.values()), create_graph=True, allow_unused=True)
        updated = {
            name: value - self.subpolicy_learning_rate * (gradient if gradient is not None else torch.zeros_like(value))
            for (name, value), gradient in zip(params.items(), gradients)
        }
        return updated, loss

    def _annealed_temperature(self) -> float:
        assert self._active_temperature_anneal_timestep is not None
        if self._active_temperature_anneal_timestep == 0:
            return 1e-8
        progress = min(1.0, self.num_timesteps / self._active_temperature_anneal_timestep)
        return max(1e-8, self.temperature * (1.0 - progress))

    def _weights(self, scores: Tensor, temperature: float) -> Tensor:
        return torch.softmax(scores / temperature, dim=0)

    def _update_performance_gains(self, scores: Tensor) -> Tensor:
        self.performance_gains.mul_(self.performance_ema_beta).add_(scores, alpha=1.0 - self.performance_ema_beta)
        return self.performance_gains

    def _mean_discounted_return(self, batch: _Batch) -> Tensor:
        rewards = batch.rewards.reshape(batch.n_steps, batch.n_envs)
        discount = torch.pow(torch.tensor(self.gamma, device=self.device), torch.arange(batch.n_steps, device=self.device))
        return (rewards * discount[:, None]).sum(dim=0).mean()

    def _select_evaluation_policy(self, params: dict[str, Tensor]) -> None:
        assert self.evaluation_policy is not None
        with torch.no_grad():
            for name, parameter in self.evaluation_policy.named_parameters():
                parameter.copy_(params[f"policy.{name}"].detach())

    def _record_selected_evaluation(self) -> None:
        if self.evaluation_env is None:
            raise ValueError("evaluation_env is required to log rollout metrics for the selected evaluation policy")
        assert self.evaluation_policy is not None
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None
        if self.get_vec_normalize_env() is not None:
            sync_envs_normalization(self.env, self.evaluation_env)
        rewards, lengths = evaluate_policy(
            self.evaluation_policy,
            self.evaluation_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        self.ep_info_buffer.clear()
        self.ep_info_buffer.extend({"r": reward, "l": length} for reward, length in zip(rewards, lengths))
        self.ep_success_buffer.clear()

    def predict(self, observation: np.ndarray, state: tuple[np.ndarray, ...] | None = None, episode_start: np.ndarray | None = None, deterministic: bool = False):
        return (self.evaluation_policy or self.policy).predict(observation, state, episode_start, deterministic)

    @staticmethod
    def _flat(tensors: tuple[Tensor, ...]) -> Tensor:
        return torch.cat([tensor.reshape(-1) for tensor in tensors])

    def _meta_update(self, loss: Tensor, observations: Tensor, old_params: dict[str, Tensor]) -> tuple[float, int, bool, float]:
        """TRPO update with the IRPO outer gradient and exact distribution KL."""
        parameters = tuple(self.policy.parameters())
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        gradient = self._flat(tuple(value if value is not None else torch.zeros_like(parameter) for parameter, value in zip(parameters, gradients))).detach()
        gradient_norm = gradient.norm().item()
        if not torch.isfinite(gradient).all() or gradient_norm == 0:
            return gradient_norm, 0, False, float("nan")
        if observations.shape[0] > self.trpo_batch_size:
            observations = observations[torch.randperm(observations.shape[0], device=self.device)[:self.trpo_batch_size]]
        frozen_old_params = {name: value.detach().clone() for name, value in old_params.items()}
        with torch.no_grad():
            old_distribution = functional_call(self._distribution_module, frozen_old_params, (observations,))

        def kl() -> Tensor:
            new_distribution = self.policy.get_distribution(observations).distribution
            return torch.distributions.kl_divergence(old_distribution, new_distribution).mean()

        def fisher_vector_product(vector: Tensor) -> Tensor:
            first = torch.autograd.grad(kl(), parameters, create_graph=True, allow_unused=True)
            flat_first = self._flat(tuple(value if value is not None else torch.zeros_like(parameter) for parameter, value in zip(parameters, first)))
            second = torch.autograd.grad((flat_first * vector).sum(), parameters, allow_unused=True)
            return self._flat(tuple(value if value is not None else torch.zeros_like(parameter) for parameter, value in zip(parameters, second))).detach() + self.trpo_damping * vector

        solution, residual, direction = torch.zeros_like(gradient), gradient.clone(), gradient.clone()
        residual_norm = torch.dot(residual, residual)
        for _ in range(self.trpo_cg_steps):
            curvature_direction = fisher_vector_product(direction)
            denominator = torch.dot(direction, curvature_direction)
            if not torch.isfinite(denominator) or denominator.abs() < 1e-12:
                break
            alpha = residual_norm / denominator
            solution += alpha * direction
            residual -= alpha * curvature_direction
            next_residual_norm = torch.dot(residual, residual)
            if not torch.isfinite(next_residual_norm) or next_residual_norm < 1e-10:
                break
            direction = residual + next_residual_norm / (residual_norm + 1e-8) * direction
            residual_norm = next_residual_norm
        curvature = 0.5 * torch.dot(solution, fisher_vector_product(solution))
        if not torch.isfinite(curvature) or curvature <= 1e-12:
            return gradient_norm, 0, False, float("nan")
        full_step = solution / torch.sqrt(curvature / self.target_kl)
        original = self._flat(tuple(parameter.detach() for parameter in parameters))

        def set_parameters(flat: Tensor) -> None:
            offset = 0
            with torch.no_grad():
                for parameter in parameters:
                    size = parameter.numel()
                    parameter.copy_(flat[offset: offset + size].reshape_as(parameter))
                    offset += size

        success, kl_value = False, float("nan")
        backtrack = self.trpo_backtrack_iters - 1
        for backtrack in range(self.trpo_backtrack_iters):
            set_parameters(original - self.trpo_backtrack_coeff**backtrack * full_step)
            kl_value = kl().item()
            if np.isfinite(kl_value) and kl_value <= self.target_kl:
                success = True
                break
        if not success:
            set_parameters(original)
        return gradient_norm, backtrack, success, kl_value

    def train(self) -> None:
        """IRPO trains inside :meth:`learn`; required by OnPolicyAlgorithm."""

    def _configure_change_output(self) -> None:
        for index, output_format in enumerate(self.logger.output_formats):
            if type(output_format) is HumanOutputFormat:
                replacement = _IRPOHumanOutputFormat(output_format.file, max(output_format.max_length, 80))
                replacement.own_file = output_format.own_file
                self.logger.output_formats[index] = replacement

    def _record_log_changes(self, iteration: int) -> None:
        assert self.ep_info_buffer is not None

        def compact(value: float) -> str:
            return (f"{value:.1e}" if 0 < abs(value) < 1e-3 else f"{value:.2g}").replace("e+", "e")

        current = {
            key: float(value)
            for key, value in self.logger.name_to_value.items()
            if key.startswith(("int_module/", "rollout/", "train/")) and isinstance(value, (float, int, np.number))
        }
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            current["rollout/ep_rew_mean"] = safe_mean([episode["r"] for episode in self.ep_info_buffer])
            current["rollout/ep_len_mean"] = safe_mean([episode["l"] for episode in self.ep_info_buffer])
        updates = iteration - self._last_log_iteration
        display_values: dict[str, str] = {}
        for key, value in current.items():
            previous = self._last_logged_values.get(key)
            if previous is None:
                continue
            delta = value - previous
            color = "\033[32m" if delta > 0 else "\033[31m" if delta < 0 else "\033[33m"
            previous_text = compact(previous)
            value_text = compact(value)
            update_text = str(updates).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
            display_values[key] = f"{color}{previous_text:>12} ──{update_text}──▶ {value_text:<12}\033[0m"
        if display_values:
            self.logger.record("_irpo_display", display_values, exclude=("stdout", "log", "csv", "tensorboard", "json"))
        self._last_logged_values = current
        self._last_log_iteration = iteration

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 10,
        tb_log_name: str = "IRPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "IRPO":
        total_timesteps, callback = self._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        callback.on_training_start(locals(), globals())
        assert self.env is not None
        self._configure_change_output()
        if reset_num_timesteps:
            self._last_log_iteration = 0
            self._last_logged_values = {}
        self._active_temperature_anneal_timestep = total_timesteps * self.temperature_anneal_timing
        iteration = 0

        while self.num_timesteps < total_timesteps:
            callback.on_rollout_start()
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            meta_params = self._params()
            # Subpolicy sample collection: meta-policy rollout, then option rollouts below.
            meta_batch = self._collect_batch(self.env, meta_params, callback)
            if meta_batch is None:
                break
            meta_intrinsic = self.intrinsic_provider.rewards(meta_batch, list(range(self.num_options)))
            option_losses: list[Tensor] = []
            scores: list[Tensor] = []
            final_params: list[dict[str, Tensor]] = []
            provider_updates: list[tuple[_Batch, int, dict[str, Tensor], Tensor]] = []
            intrinsic_critic_losses: list[float] = []
            extrinsic_critic_losses: list[float] = []
            complete = True
            for option in range(self.num_options):
                params, batch = meta_params, meta_batch
                final_batch: _Batch | None = None
                final_loss: Tensor | None = None
                final_external_advantages: Tensor | None = None
                for update in range(self.num_subpolicy_updates):
                    if update:
                        batch = self._collect_batch(self.env, params, callback)
                        if batch is None:
                            complete = False
                            break
                        intrinsic_rewards = self.intrinsic_provider.rewards(batch, [option])[option]
                    else:
                        intrinsic_rewards = meta_intrinsic[option]
                    advantages, returns = self.option_critics.targets(batch, option, intrinsic_rewards, intrinsic=True)
                    intrinsic_critic_losses.append(
                        self.option_critics.update(batch, option, intrinsic_rewards, intrinsic=True, returns=returns)
                    )
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
                    # N differentiable intrinsic-reward adaptations: theta_1 -> theta_{N+1}.
                    params, _ = self._adapt(params, batch, advantages)
                if not complete:
                    break
                # The selected policy is theta_{N+1}. Its external rollout is a
                # fresh on-policy outer batch, rather than data from theta_N.
                final_batch = self._collect_batch(self.env, params, callback)
                if final_batch is None:
                    complete = False
                    break
                final_external_advantages, returns = self.option_critics.targets(
                    final_batch, option, final_batch.rewards, intrinsic=False
                )
                extrinsic_critic_losses.append(
                    self.option_critics.update(final_batch, option, final_batch.rewards, intrinsic=False, returns=returns)
                )
                normalized_external_advantages = (final_external_advantages - final_external_advantages.mean()) / (
                    final_external_advantages.std(unbiased=False) + 1e-8
                )
                final_loss = self._policy_loss(params, final_batch, normalized_external_advantages)
                if not complete or final_batch is None or final_loss is None or final_external_advantages is None:
                    break
                # Meta-policy update uses the gradient of this final external update,
                # matching the research backpropagation topology.
                option_losses.append(final_loss)
                scores.append(self._mean_discounted_return(final_batch).detach())
                final_params.append(params)
                # Source providers receive the final cloned subpolicy, not the
                # retained IRPO meta-gradient graph.
                provider_params = {
                    name: value.detach().clone().requires_grad_(value.requires_grad)
                    for name, value in params.items()
                }
                provider_updates.append((final_batch, option, provider_params, final_external_advantages))
            callback.on_rollout_end()
            if not complete:
                break

            intrinsic_logs: dict[str, list[float]] = {}
            for final_batch, option, params, external_advantages in provider_updates:
                for name, value in self.intrinsic_provider.update(
                    final_batch, option, policy_evaluate=self._evaluate_actions, params=params,
                    subpolicy_learning_rate=self.subpolicy_learning_rate, external_advantages=external_advantages,
                ).items():
                    intrinsic_logs.setdefault(name, []).append(value)
            score_tensor = torch.stack(scores)
            performance_gains = self._update_performance_gains(score_tensor)
            temperature = self._annealed_temperature()
            weights = self._weights(performance_gains, temperature)
            selected_option = performance_gains.argmax().item()
            # Evaluation policy selection happens on every completed IRPO update loop.
            self._select_evaluation_policy(final_params[selected_option])
            meta_loss = torch.sum(torch.stack(option_losses) * weights)
            # Meta-policy update: apply the option-aggregated outer gradient to SB3 policy.
            gradient_norm, backtrack, trpo_success, trpo_kl = self._meta_update(meta_loss, meta_batch.observations, meta_params)
            iteration += 1
            self._n_updates += 1
            self.logger.record("train/meta_loss", meta_loss.item())
            if self._allo_pretraining_steps:
                self.logger.record("train/allo_pretraining_timesteps", self._allo_pretraining_steps)
            self.logger.record("train/meta_gradient_norm", gradient_norm)
            # self.logger.record("train/trpo_backtrack", backtrack)
            # self.logger.record("train/trpo_success", trpo_success)
            self.logger.record("train/meta-policy kl", trpo_kl)
            if intrinsic_critic_losses:
                self.logger.record("train/intrinsic_critic_loss", np.mean(intrinsic_critic_losses))
            if extrinsic_critic_losses:
                self.logger.record("train/extrinsic_critic_loss", np.mean(extrinsic_critic_losses))
            for name, values in intrinsic_logs.items():
                self.logger.record(f"int_module/{name}", np.mean(values))
            self.logger.record("train/option_return", score_tensor[selected_option].item())
            self.logger.record("train/option_ema_return", performance_gains[selected_option].item())
            # self.logger.record("train/option_weight_max", weights.max().item())
            self.logger.record("train/selected_option", selected_option)
            self.logger.record("train/temperature", temperature)
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            if log_interval and iteration % log_interval == 0:
                self._record_selected_evaluation()
                self._record_log_changes(iteration)
                self.dump_logs(iteration)

        callback.on_training_end()
        return self

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        return ["policy", "policy.optimizer", "intrinsic_provider", "option_critics", "evaluation_policy"], ["performance_gains"]

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "evaluation_env", "_last_log_iteration", "_last_logged_values"]
