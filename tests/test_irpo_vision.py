"""Vision-provider check for the 4x84x84 Atari path."""

from types import SimpleNamespace

import torch
from gymnasium import spaces

from stable_baselines3.irpo.intrinsic import ALLOReward, DRNDReward, LIRPGReward, RandomReward


def batch() -> SimpleNamespace:
    return SimpleNamespace(
        observations=torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8),
        next_observations=torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8),
        actions=torch.tensor([0, 1, 2, 3]),
        rewards=torch.ones(4),
        terminations=torch.zeros(4),
        truncations=torch.ones(4),
        log_probs=torch.zeros(4),
        n_steps=1,
        n_envs=4,
    )


def main() -> None:
    torch.set_num_threads(1)
    observation_space = spaces.Box(0, 255, (4, 84, 84), dtype="uint8")
    action_space = spaces.Discrete(4)
    data = batch()
    random = RandomReward(observation_space, 3)
    assert torch.isfinite(random.rewards(data, [0, 1, 2])[0]).all()
    drnd = DRNDReward(observation_space, 2, 3e-5, 0.99, 0.95, feature_dim=16)
    targets = [parameter.detach().clone() for slot in drnd.slots for target in slot.drnd.target for parameter in target.parameters()]
    drnd.update(data, 0)
    assert torch.isfinite(drnd.rewards(data, [0, 1])[1]).all()
    current = [parameter for slot in drnd.slots for target in slot.drnd.target for parameter in target.parameters()]
    assert all(torch.equal(before, after) for before, after in zip(targets, current))
    lirpg = LIRPGReward(observation_space, action_space, 2, 1e-4, 0.99, 0.95)
    assert torch.isfinite(lirpg.rewards(data, [0, 1])[0]).all()
    allo = ALLOReward(observation_space, 3, 1e-4)
    barrier_before = allo.barrier.clone()
    allo.pretrain(data.observations[None], data.next_observations[None], torch.zeros(1, 4, dtype=torch.bool), updates=1, batch_size=4)
    assert torch.isfinite(allo.rewards(data, [0, 1, 2])[2]).all()
    assert torch.all(allo.barrier >= barrier_before) and allo.lr_scheduler.last_epoch == 1
    assert any(isinstance(module, torch.nn.Conv2d) for module in drnd.modules())


if __name__ == "__main__":
    main()
