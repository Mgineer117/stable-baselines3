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
from stable_baselines3 import IRPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = VecFrameStack(make_atari_env("ALE/Pacman-v5", n_envs=1), n_stack=4)
evaluation_env = VecFrameStack(make_atari_env("ALE/Pacman-v5", n_envs=1), n_stack=4)
model = IRPO(
    "CnnPolicy",
    env,
    evaluation_env=evaluation_env,
    intrinsic_reward="drnd",  # random | allo | lirpg | drnd
    num_options=3,
    num_subpolicy_updates=5,
    subpolicy_learning_rate=1e-4,
    drnd_learning_rate=3e-5,
    drnd_feature_dim=16,
    n_steps=2048,
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
`num_subpolicy_updates` differentiable intrinsic-reward updates per option,
then collects a fresh external rollout from the resulting final subpolicy. The
external policy-gradient loss from that final-policy rollout is differentiated
through all inner updates for the TRPO meta update. IRPO selects the final
subpolicy with the highest exponential-moving-average (EMA) of its mean
discounted final-rollout return for `predict()`. The EMA follows ASDF's
`gain = beta * gain + (1 - beta) * return`, with `performance_ema_beta=0.9` by
default. Selection happens after every completed IRPO outer update. `rollout/ep_rew_mean`
and `rollout/ep_len_mean` are the mean undiscounted return and length of five
completed deterministic episodes from that selected policy. Pass a separate
`evaluation_env`; it must not be the training environment.

IRPO owns a distinct intrinsic and extrinsic critic for every option. Intrinsic
critics update with each intrinsic subpolicy batch; after all subpolicy updates,
the corresponding extrinsic critic updates once from the fresh final-policy
batch. Both use PPO-style fixed GAE targets, shuffled minibatches, and complete
passes over the rollout (`critic_batch_size=64`, `critic_n_epochs=10`).
`critic_learning_rate=1e-3` controls both banks.

## Intrinsic rewards

- `random` uses fixed signed temporal feature differences and a reward RMS.
- `drnd` uses one ten-target DRND ensemble, novelty RMS, and masked predictor
  updates.
- `lirpg` uses an action-conditioned learned reward and the source virtual
  clipped-policy update. Its default coefficients are
  `lirpg_r_ex_coef=1.0` and `lirpg_r_in_coef=0.01`.
- `allo` pretrains its temporal-eigenfunction encoder during `IRPO`
  construction. It uses 100,000 random image transitions or 200,000 vector
  transitions and 10,000 ALLO updates by default. Reduce those only for a
  smoke run, for example:

```python
model = IRPO(
    "MlpPolicy",
    env,
    evaluation_env=evaluation_env,
    intrinsic_reward="allo",
    allo_pretrain_timesteps=2_000,
    allo_pretrain_updates=100,
)
```

`allo_learning_rate`, `drnd_learning_rate`, and `lirpg_learning_rate` control
their corresponding providers. Provider weights, normalizers, and optimizer
state are saved with `model.save()`.

Learned-provider diagnostics are logged under `int_module/` (displayed directly
below `rollout/` in the terminal): DRND logs
`loss`, `predictor_loss`, and `intrinsic_reward_mean`; LIRPG logs `loss`,
`meta_loss`, and `intrinsic_reward_mean`. Fixed Random and frozen ALLO providers
have no invented learning-loss logs.
Shared policy-critic losses are logged as `train/intrinsic_critic_loss` and
`train/extrinsic_critic_loss`.

IRPO dumps metrics every ten outer updates by default. Its terminal values for
rollout, train, and intrinsic-reward metrics become
`previous ──¹⁰──▶ current`; positive transitions are green and negative ones
red. The 80-column terminal table has visible `+` corners and fixed-width
endpoints, so it remains rectangular and every arrow aligns horizontally. CSV
and TensorBoard retain only raw numeric metrics. Unless redirected with
`tensorboard_log=...`, IRPO always writes TensorBoard events below
`./tensorboard/IRPO_*`.

## Deliberate SB3 differences

Each option rollout is collected serially from an SB3 `VecEnv`. This is an
intentional SB3 choice and differs from ASDF's parallel sampling.

Softmax aggregation is always used. `temperature_anneal_timing` is the fraction
of `learn(total_timesteps=...)` when its temperature reaches the argmax limit:
`0.1` means 10% of the requested budget, and `0` is argmax immediately.
`train/option_return` is the selected option's current `n_steps` discounted
sample score; `train/option_ema_return` is the EMA that selects and weights options.
