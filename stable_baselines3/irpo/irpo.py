"""Intrinsic Reward Policy Optimization for Stable-Baselines3."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces
from torch import Tensor, nn
from torch.func import functional_call

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
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
        n_steps: int = 128,
        gamma: float = 0.99,
        ent_coef: float = 0.0,
        # Subpolicy hyperparameters.
        num_options: int = 3,
        subpolicy_learning_rate: float = 3e-4,
        num_subpolicy_updates: int = 5,
        # Intrinsic-reward hyperparameters.
        intrinsic_reward: IntrinsicReward = "random",
        lirpg_learning_rate: float = 7e-4,
        drnd_learning_rate: float = 1e-4,
        allo_learning_rate: float = 3e-4,
        allo_encoder_path: str | None = None,
        # Meta-policy hyperparameters.
        temperature: float = 1.0,
        temperature_anneal_timing: float = 1.0,
        target_kl: float = 0.001,
        trpo_damping: float = 0.1,
        trpo_cg_steps: int = 5,
        trpo_backtrack_iters: int = 10,
        trpo_backtrack_coeff: float = 0.7,
        # Logging and device.
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
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= temperature_anneal_timing <= 1.0:
            raise ValueError("temperature_anneal_timing must be in [0, 1]")
        if target_kl <= 0 or trpo_damping < 0 or trpo_cg_steps < 1 or trpo_backtrack_iters < 1:
            raise ValueError("invalid TRPO hyperparameters")
        if not isinstance(env, str) and isinstance(env.observation_space, spaces.Dict):
            raise NotImplementedError("The initial IRPO port supports Box observations only")

        super().__init__(
            policy,
            env,
            # SB3 creates a policy optimizer, but IRPO updates the base policy with TRPO.
            learning_rate=0.0,
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
        self.subpolicy_learning_rate = subpolicy_learning_rate
        self.lirpg_learning_rate = lirpg_learning_rate
        self.temperature = temperature
        self.temperature_anneal_timing = temperature_anneal_timing
        self._active_temperature_anneal_timestep: float | None = None
        self.target_kl = target_kl
        self.trpo_damping = trpo_damping
        self.trpo_cg_steps = trpo_cg_steps
        self.trpo_backtrack_iters = trpo_backtrack_iters
        self.trpo_backtrack_coeff = trpo_backtrack_coeff
        self.intrinsic_reward_kind = intrinsic_reward
        self.allo_encoder_path = allo_encoder_path
        self.drnd_learning_rate = drnd_learning_rate
        self.allo_learning_rate = allo_learning_rate
        self._lirpg_optimizer: torch.optim.Optimizer | None = None
        self.evaluation_policy: ActorCriticPolicy | None = None

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
        self.evaluation_policy = deepcopy(self.policy)

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
        return {
            name: value - self.subpolicy_learning_rate * (gradient if gradient is not None else torch.zeros_like(value))
            for (name, value), gradient in zip(params.items(), gradients)
        }

    def _annealed_temperature(self) -> float:
        assert self._active_temperature_anneal_timestep is not None
        if self._active_temperature_anneal_timestep == 0:
            return 1e-8
        progress = min(1.0, self.num_timesteps / self._active_temperature_anneal_timestep)
        return max(1e-8, self.temperature * (1.0 - progress))

    def _weights(self, scores: Tensor, temperature: float) -> Tensor:
        return torch.softmax(scores / temperature, dim=0)

    def _mean_discounted_return(self, batch: _Batch) -> Tensor:
        rewards = batch.rewards.reshape(batch.n_steps, batch.n_envs)
        discount = torch.pow(torch.tensor(self.gamma, device=self.device), torch.arange(batch.n_steps, device=self.device))
        return (rewards * discount[:, None]).sum(dim=0).mean()

    def _select_evaluation_policy(self, params: dict[str, Tensor]) -> None:
        assert self.evaluation_policy is not None
        with torch.no_grad():
            for name, parameter in self.evaluation_policy.named_parameters():
                parameter.copy_(params[f"policy.{name}"].detach())

    def predict(self, observation: np.ndarray, state: tuple[np.ndarray, ...] | None = None, episode_start: np.ndarray | None = None, deterministic: bool = False):
        policy = self.evaluation_policy or self.policy
        return policy.predict(observation, state, episode_start, deterministic)

    def _update_lirpg(self, loss: Tensor) -> None:
        if not isinstance(self.intrinsic_provider, LIRPGReward):
            return
        parameters = tuple(self.intrinsic_provider.parameters())
        if self._lirpg_optimizer is None:
            self._lirpg_optimizer = torch.optim.RMSprop(parameters, lr=self.lirpg_learning_rate)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        self._lirpg_optimizer.zero_grad()
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = None if gradient is None else gradient.detach()
        self._lirpg_optimizer.step()

    @staticmethod
    def _flat(tensors: tuple[Tensor, ...]) -> Tensor:
        return torch.cat([tensor.reshape(-1) for tensor in tensors])

    def _meta_update(self, loss: Tensor, observations: Tensor) -> tuple[float, int, bool, float]:
        """TRPO update using the IRPO outer gradient as the CG right-hand side."""
        parameters = tuple(self.policy.parameters())
        raw_gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        gradients = tuple(
            gradient if gradient is not None else torch.zeros_like(parameter)
            for parameter, gradient in zip(parameters, raw_gradients)
        )
        gradient = self._flat(gradients).detach()
        gradient_norm = gradient.norm().item()
        if not torch.isfinite(gradient).all() or gradient_norm == 0:
            return gradient_norm, 0, False, float("nan")

        if observations.shape[0] > 256:
            observations = observations[torch.randperm(observations.shape[0], device=self.device)[:256]]
        with torch.no_grad():
            old_distribution = self.policy.get_distribution(observations)
            old_actions = old_distribution.get_actions(deterministic=False)
            old_log_prob = old_distribution.log_prob(old_actions)

        def kl() -> Tensor:
            distribution = self.policy.get_distribution(observations)
            log_ratio = distribution.log_prob(old_actions) - old_log_prob
            return ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()

        def fisher_vector_product(vector: Tensor) -> Tensor:
            first = torch.autograd.grad(kl(), parameters, create_graph=True, allow_unused=True)
            flat_first = self._flat(tuple(
                value if value is not None else torch.zeros_like(parameter)
                for parameter, value in zip(parameters, first)
            ))
            second = torch.autograd.grad((flat_first * vector).sum(), parameters, allow_unused=True)
            return self._flat(tuple(
                value if value is not None else torch.zeros_like(parameter)
                for parameter, value in zip(parameters, second)
            )).detach() + self.trpo_damping * vector

        solution = torch.zeros_like(gradient)
        residual = gradient.clone()
        direction = gradient.clone()
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
                    parameter.copy_(flat[offset : offset + size].reshape_as(parameter))
                    offset += size

        success = False
        kl_value = float("nan")
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
        self._active_temperature_anneal_timestep = total_timesteps * self.temperature_anneal_timing
        iteration = 0

        while self.num_timesteps < total_timesteps:
            callback.on_rollout_start()
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            base_params = self._params()
            # Subpolicy sample collection: base rollout, then option rollouts below.
            base_batch = self._collect_batch(self.env, base_params, callback)
            if base_batch is None:
                break
            option_losses: list[Tensor] = []
            scores: list[Tensor] = []
            final_params: list[dict[str, Tensor]] = []
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
                    # Subpolicy update: differentiable intrinsic/extrinsic policy-gradient step.
                    params = self._adapt(params, batch, option, final)
                    if final:
                        final_batch = batch
                if not complete or final_batch is None:
                    break
                external_advantage = self._returns(
                    final_batch.rewards, final_batch.dones, final_batch.n_steps, final_batch.n_envs
                )
                option_losses.append(self._policy_loss(params, final_batch, external_advantage))
                scores.append(self._mean_discounted_return(final_batch).detach())
                final_params.append(params)
                self.intrinsic_provider.update(final_batch.observations, final_batch.next_observations)
            callback.on_rollout_end()
            if not complete:
                break

            score_tensor = torch.stack(scores)
            temperature = self._annealed_temperature()
            weights = self._weights(score_tensor, temperature)
            selected_option = score_tensor.argmax().item()
            self._select_evaluation_policy(final_params[selected_option])
            meta_loss = torch.sum(torch.stack(option_losses) * weights)
            self._update_lirpg(meta_loss)
            # Meta-policy update: apply the option-aggregated outer gradient to SB3 policy.
            gradient_norm, backtrack, trpo_success, trpo_kl = self._meta_update(meta_loss, base_batch.observations)
            iteration += 1
            self._n_updates += 1
            self.logger.record("train/meta_loss", meta_loss.item())
            self.logger.record("train/meta_gradient_norm", gradient_norm)
            self.logger.record("train/trpo_backtrack", backtrack)
            self.logger.record("train/trpo_success", trpo_success)
            self.logger.record("train/trpo_kl", trpo_kl)
            self.logger.record("train/option_return", score_tensor.mean().item())
            self.logger.record("train/option_weight_max", weights.max().item())
            self.logger.record("train/selected_option", selected_option)
            self.logger.record("train/temperature", temperature)
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            if log_interval and iteration % log_interval == 0:
                self.dump_logs(iteration)

        callback.on_training_end()
        return self

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        return ["policy", "intrinsic_provider", "evaluation_policy"], []
