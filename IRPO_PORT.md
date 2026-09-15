# IRPO port

IRPO will be added as `stable_baselines3.IRPO` while retaining SB3's policy,
VecEnv, callback, logging, save/load, and Gymnasium interfaces.

The implementation must replace ASDF's `OnlineSampler` and `OnPolicyTrainer`.
They are the only two repository-specific execution layers; copying them into
SB3 would make the fork harder to install and maintain.

## Public interface

```python
from stable_baselines3 import IRPO

model = IRPO(
    "CnnPolicy",
    env,
    intrinsic_reward="drnd",  # random, allo, lirpg, drnd
    num_options=3,
    num_subpolicy_updates=5,
)
model.learn(100_000_000)
```

`allo` requires an explicit pretrained encoder checkpoint supplied as
`allo_encoder_path`. Pretraining is an explicit command, never a hidden phase
inside `learn()`.

## Port order

1. IRPO rollout and differentiable subpolicy updates on SB3 `VecEnv`.
2. The Random provider and a CartPole integration test.
3. DRND, LIRPG, and ALLO provider ports with their checkpoint state.
4. Atari and MuJoCo parity checks against the existing implementation.
