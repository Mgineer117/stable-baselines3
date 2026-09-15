"""Small executable check for the SB3-native IRPO path."""

from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import torch

from stable_baselines3 import IRPO
from stable_baselines3.irpo.intrinsic import _RewardNetwork


def train(kind: str, **kwargs: str) -> None:
    model = IRPO(
        "MlpPolicy",
        gym.make("CartPole-v1"),
        intrinsic_reward=kind,  # type: ignore[arg-type]
        num_options=2,
        num_subpolicy_updates=2,
        n_steps=4,
        subpolicy_learning_rate=1e-3,
        device="cpu",
        seed=0,
        **kwargs,
    )
    model.learn(32)
    assert model.num_timesteps >= 32


def main() -> None:
    for kind in ("random", "drnd", "lirpg"):
        train(kind)

    with TemporaryDirectory() as directory:
        encoder = _RewardNetwork(2)
        encoder(torch.zeros(1, 4))
        path = Path(directory) / "allo.pt"
        torch.save({"encoder": encoder.state_dict()}, path)
        train("allo", allo_encoder_path=str(path))


if __name__ == "__main__":
    main()
