"""Run: PYTHONPATH=. /home/minjae/miniconda3/envs/irpo/bin/python tests/test_irpo_smoke.py"""

from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import torch

from stable_baselines3 import IRPO
from stable_baselines3.irpo.intrinsic import DRNDReward, LIRPGReward


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
    selected = 0
    original = model._select_evaluation_policy

    def record(params: dict[str, torch.Tensor]) -> None:
        nonlocal selected
        selected += 1
        original(params)

    model._select_evaluation_policy = record  # type: ignore[method-assign]
    target_before = None
    if isinstance(model.intrinsic_provider, DRNDReward):
        target_before = [parameter.detach().clone() for slot in model.intrinsic_provider.slots for target in slot.drnd.target for parameter in target.parameters()]
    model.learn(32)
    assert model.num_timesteps >= 32
    assert selected == model._n_updates and selected > 1
    assert model.evaluation_policy is not None
    if target_before is not None:
        target_after = [parameter for slot in model.intrinsic_provider.slots for target in slot.drnd.target for parameter in target.parameters()]
        assert all(torch.equal(before, after) for before, after in zip(target_before, target_after))
    return model


def check_resume(kind: str) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / kind
        model = train(kind)
        model.save(path)
        restored = IRPO.load(path, env=gym.make("CartPole-v1"), device="cpu")
        assert isinstance(restored.intrinsic_provider, (DRNDReward, LIRPGReward))
        provider = restored.intrinsic_provider
        assert _optimizer_parameter_ids(provider) == {id(parameter) for parameter in provider.parameters()}
        restored.learn(16)


def main() -> None:
    for kind in ("random", "allo", "drnd", "lirpg"):
        train(kind)
    for kind in ("drnd", "lirpg"):
        check_resume(kind)


if __name__ == "__main__":
    main()
