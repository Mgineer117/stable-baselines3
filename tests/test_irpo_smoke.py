"""Run: PYTHONPATH=. /home/minjae/miniconda3/envs/irpo/bin/python tests/test_irpo_smoke.py"""

import csv
import re
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import torch

from stable_baselines3 import IRPO
from stable_baselines3.irpo.intrinsic import DRNDReward, LIRPGReward
from stable_baselines3.common.logger import HumanOutputFormat, Logger, configure
from stable_baselines3.common.monitor import Monitor


def _optimizer_parameter_ids(provider: DRNDReward | LIRPGReward) -> set[int]:
    return {
        id(parameter)
        for slot in provider.slots
        for group in slot.optimizer.param_groups
        for parameter in group["params"]
    }


def train(kind: str) -> IRPO:
    kwargs = {"allo_pretrain_timesteps": 8, "allo_pretrain_updates": 1} if kind == "allo" else {}
    model = IRPO(
        "MlpPolicy",
        gym.make("CartPole-v1"),
        evaluation_env=Monitor(gym.make("CartPole-v1")),
        intrinsic_reward=kind,  # type: ignore[arg-type]
        num_options=2,
        num_subpolicy_updates=2,
        n_steps=4,
        subpolicy_learning_rate=1e-3,
        trpo_batch_size=4,
        device="cpu",
        seed=0,
        **kwargs,
    )
    intrinsic_before = [parameter.detach().clone() for critic in model.option_critics.intrinsic for parameter in critic.parameters()]
    extrinsic_before = [parameter.detach().clone() for critic in model.option_critics.extrinsic for parameter in critic.parameters()]
    if isinstance(model.intrinsic_provider, (DRNDReward, LIRPGReward)):
        assert all(not hasattr(slot, "critic") for slot in model.intrinsic_provider.slots)
    selected = 0
    original = model._select_evaluation_policy
    critic_update_kinds: list[bool] = []
    original_critic_update = model.option_critics.update
    collected = 0
    original_collect = model._collect_batch

    def record(params: dict[str, torch.Tensor]) -> None:
        nonlocal selected
        selected += 1
        original(params)

    def record_critic_update(batch, option, rewards, intrinsic, **kwargs):
        critic_update_kinds.append(intrinsic)
        return original_critic_update(batch, option, rewards, intrinsic)

    def record_collect(env, params, callback):
        nonlocal collected
        collected += 1
        return original_collect(env, params, callback)

    model._select_evaluation_policy = record  # type: ignore[method-assign]
    model.option_critics.update = record_critic_update  # type: ignore[method-assign]
    model._collect_batch = record_collect  # type: ignore[method-assign]
    target_before = None
    if isinstance(model.intrinsic_provider, DRNDReward):
        target_before = [parameter.detach().clone() for slot in model.intrinsic_provider.slots for target in slot.drnd.target for parameter in target.parameters()]
    model.learn(32)
    assert model.num_timesteps >= 32
    assert selected == model._n_updates and selected > 1
    # One shared meta rollout, then N per-option rollouts including the final-policy external batch.
    assert collected == model._n_updates * (1 + model.num_options * model.num_subpolicy_updates)
    assert critic_update_kinds.count(False) == model._n_updates * model.num_options
    assert critic_update_kinds.count(True) == model._n_updates * model.num_options * model.num_subpolicy_updates
    assert model.evaluation_policy is not None
    intrinsic_after = [parameter for critic in model.option_critics.intrinsic for parameter in critic.parameters()]
    extrinsic_after = [parameter for critic in model.option_critics.extrinsic for parameter in critic.parameters()]
    assert any(not torch.equal(before, after) for before, after in zip(intrinsic_before, intrinsic_after))
    assert any(not torch.equal(before, after) for before, after in zip(extrinsic_before, extrinsic_after))
    if target_before is not None:
        target_after = [parameter for slot in model.intrinsic_provider.slots for target in slot.drnd.target for parameter in target.parameters()]
        assert all(torch.equal(before, after) for before, after in zip(target_before, target_after))
    # Do not serialize test-only closures into the checkpoint-resume test.
    del model._select_evaluation_policy
    del model.option_critics.update
    del model._collect_batch
    return model


def check_resume(kind: str) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / kind
        model = train(kind)
        model.save(path)
        restored = IRPO.load(
            path,
            env=gym.make("CartPole-v1"),
            evaluation_env=Monitor(gym.make("CartPole-v1")),
            device="cpu",
        )
        assert isinstance(restored.intrinsic_provider, (DRNDReward, LIRPGReward))
        assert torch.allclose(restored.performance_gains, model.performance_gains)
        assert torch.allclose(restored.option_critics.intrinsic[0].value.weight, model.option_critics.intrinsic[0].value.weight)
        assert restored.get_parameters()["policy.optimizer"] == model.get_parameters()["policy.optimizer"]
        assert restored.option_critics.intrinsic_optimizers[0].state
        assert restored.option_critics.extrinsic_optimizers[0].state
        provider = restored.intrinsic_provider
        assert _optimizer_parameter_ids(provider) == {id(parameter) for parameter in provider.parameters()}
        before = restored.num_timesteps
        restored.learn(16, reset_num_timesteps=False)
        assert restored.num_timesteps >= before + 16


def check_option_ema() -> None:
    model = IRPO(
        "MlpPolicy",
        gym.make("CartPole-v1"),
        evaluation_env=Monitor(gym.make("CartPole-v1")),
        num_options=2,
        num_subpolicy_updates=2,
        performance_ema_beta=0.5,
        device="cpu",
    )
    assert torch.allclose(model._update_performance_gains(torch.tensor([4.0, 2.0])), torch.tensor([2.0, 1.0]))
    assert torch.allclose(model._update_performance_gains(torch.tensor([2.0, 6.0])), torch.tensor([2.0, 3.5]))


def check_rollout_logging(kind: str, intrinsic_fields: set[str]) -> None:
    with TemporaryDirectory() as directory:
        model = IRPO(
            "MlpPolicy",
            gym.make("CartPole-v1", max_episode_steps=2),
            evaluation_env=Monitor(gym.make("CartPole-v1", max_episode_steps=8)),
            n_eval_episodes=2,
            intrinsic_reward=kind,  # type: ignore[arg-type]
            num_options=2,
            num_subpolicy_updates=2,
            n_steps=4,
            trpo_batch_size=4,
            device="cpu",
            seed=0,
        )
        model.set_logger(configure(directory, ["csv"]))
        model.learn(64, log_interval=1)
        with (Path(directory) / "progress.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        assert rows
        assert {
            "rollout/ep_rew_mean", "rollout/ep_len_mean", "time/fps", "train/option_ema_return",
            "train/intrinsic_critic_loss", "train/extrinsic_critic_loss",
        } <= set(rows[0])
        assert intrinsic_fields <= set(rows[0])
        assert "_irpo_display" not in rows[0]
        assert all(float(row["rollout/ep_rew_mean"]) == 8 for row in rows)
        assert all(float(row["rollout/ep_len_mean"]) == 8 for row in rows)


def check_log_changes() -> None:
    output = StringIO()
    model = IRPO(
        "MlpPolicy",
        gym.make("CartPole-v1", max_episode_steps=2),
        evaluation_env=Monitor(gym.make("CartPole-v1", max_episode_steps=8)),
        n_eval_episodes=1,
        intrinsic_reward="drnd",
        num_options=2,
        num_subpolicy_updates=2,
        n_steps=4,
        trpo_batch_size=4,
        device="cpu",
        seed=0,
    )
    model.set_logger(Logger("", [HumanOutputFormat(output)]))
    model.learn(80, log_interval=2)
    assert "change/" not in output.getvalue()
    assert "int_module/" in output.getvalue()
    assert output.getvalue().index("| rollout/") < output.getvalue().index("| int_module/")
    assert "──²──▶" in output.getvalue()
    assert "\033[" in output.getvalue()
    plain_lines = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in output.getvalue().splitlines()]
    borders = [index for index, line in enumerate(plain_lines) if line.startswith("+") and line.endswith("+")]
    assert borders
    for start, end in zip(borders[::2], borders[1::2]):
        assert all(len(line) == len(plain_lines[start]) for line in plain_lines[start : end + 1])
    transition_lines = [
        line for line in plain_lines if "──²──▶" in line
    ]
    assert len({line.index("──²──▶") for line in transition_lines}) == 1


def main() -> None:
    for kind in ("random", "allo", "drnd", "lirpg"):
        train(kind)
    for kind in ("drnd", "lirpg"):
        check_resume(kind)
    check_option_ema()
    check_rollout_logging("drnd", {"int_module/loss", "int_module/predictor_loss", "int_module/intrinsic_reward_mean"})
    check_rollout_logging("lirpg", {"int_module/loss", "int_module/meta_loss", "int_module/intrinsic_reward_mean"})
    check_log_changes()


if __name__ == "__main__":
    main()
