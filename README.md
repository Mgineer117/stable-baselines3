# Stable-Baselines3 IRPO

This fork adds `stable_baselines3.IRPO`: Intrinsic Reward Policy Optimization
with `random`, `allo`, `lirpg`, and `drnd` intrinsic-reward providers.

## Install

```bash
pip install --upgrade \
  "stable-baselines3 @ git+https://github.com/Mgineer117/stable-baselines3.git@irpo"
```

## Pacman

```python
import gymnasium as gym
from stable_baselines3 import IRPO

env = gym.make("ALE/Pacman-v5")
model = IRPO(
    "CnnPolicy",
    env,
    intrinsic_reward="drnd",  # random | allo | lirpg | drnd
    num_options=3,
    num_subpolicy_updates=5,
    subpolicy_learning_rate=1e-4,
    drnd_learning_rate=3e-5,
    drnd_feature_dim=16,
    n_steps=128,
    clip_training_rewards=True,
    temperature_anneal_timing=0.5,
    verbose=1,
)
model.learn(100_000_000)
model.save("irpo_drnd_pacman")
```

Use `MlpPolicy` for vector observations and `CnnPolicy` for channel-first image
observations. Intrinsic image providers use their own Nature-style CNNs; they
do not flatten Atari frames into an MLP.

Each IRPO outer update collects a meta-policy rollout, performs
`num_subpolicy_updates` differentiable updates per option, uses the gradient of
the final external subpolicy update for the TRPO meta update, then selects the
final subpolicy with the highest mean discounted return for `predict()`.
Selection happens after every completed IRPO outer update.

## Intrinsic rewards

- `random` uses fixed signed temporal feature differences and a reward RMS.
- `drnd` uses one ten-target DRND ensemble, novelty RMS, masked predictor
  updates, and a provider-local GAE critic for each option.
- `lirpg` uses an action-conditioned learned reward, an external critic, and
  the source virtual clipped-policy update. Its default coefficients are
  `lirpg_r_ex_coef=1.0`, `lirpg_r_in_coef=0.01`, and
  `lirpg_v_ex_coef=0.5`.
- `allo` pretrains its temporal-eigenfunction encoder during `IRPO`
  construction. It uses 100,000 random image transitions or 200,000 vector
  transitions and 10,000 ALLO updates by default. Reduce those only for a
  smoke run, for example:

```python
model = IRPO(
    "MlpPolicy",
    env,
    intrinsic_reward="allo",
    allo_pretrain_timesteps=2_000,
    allo_pretrain_updates=100,
)
```

`allo_learning_rate`, `drnd_learning_rate`, and `lirpg_learning_rate` control
their corresponding providers. Provider weights, normalizers, and optimizer
state are saved with `model.save()`.

## Deliberate SB3 differences

The core IRPO inner update uses normalized discounted reward-to-go rather than
research IRPO's per-option intrinsic/extrinsic critics. Each option rollout is
collected serially from an SB3 `VecEnv` and begins with `reset()`. These are
intentional SB3 choices; they make this implementation a distinct variant for
those two mechanisms.

Softmax aggregation is always used. `temperature_anneal_timing` is the fraction
of `learn(total_timesteps=...)` when its temperature reaches the argmax limit:
`0.1` means 10% of the requested budget, and `0` is argmax immediately.
