# Stable-Baselines3 IRPO

This fork adds **Intrinsic Reward Policy Optimization (IRPO)** to
Stable-Baselines3. IRPO owns its intrinsic-reward providers:
`random`, `allo`, `lirpg`, and `drnd`.

> **Status:** The first SB3-native IRPO implementation supports Box observations,
> `MlpPolicy`/`CnnPolicy`, and SGD meta updates. Atari and MuJoCo parity with the
> research implementation are still being validated.

## Install

Install this fork in place of upstream Stable-Baselines3:

```bash
pip install --upgrade \
  "stable-baselines3 @ git+https://github.com/Mgineer117/stable-baselines3.git@irpo"
```

## Use IRPO

```python
import gymnasium as gym
from stable_baselines3 import IRPO

env = gym.make("ALE/MsPacman-v5")
model = IRPO(
    "CnnPolicy",
    env,
    intrinsic_reward="drnd",  # random | allo | lirpg | drnd
    num_options=3,
    num_subpolicy_updates=5,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=128,
    verbose=1,
)
model.learn(100_000_000)
model.save("irpo_drnd_pacman")
```

Use `"MlpPolicy"` for vector observations and `"CnnPolicy"` for images.
`num_options` selects the number of IRPO subpolicies; each iteration makes
`num_subpolicy_updates` differentiable updates per option.

This SB3 port deliberately uses normalized discounted returns as the policy-gradient
baseline. It does not train separate intrinsic and extrinsic critics.

## Intrinsic reward

- `random`: fixed random reward functions; no pretraining.
- `drnd`: discounted random-network-distillation reward; no pretraining.
- `lirpg`: learned intrinsic reward; no pretraining.
- `allo`: frozen ALLO encoder. Pretrain the encoder, then pass its checkpoint:

```python
model = IRPO(
    "CnnPolicy",
    env,
    intrinsic_reward="allo",
    allo_encoder_path="allo_encoder.pt",
)
```

`allo_encoder_path` is required only for ALLO. Saving an IRPO model preserves
the selected intrinsic provider and its state.
