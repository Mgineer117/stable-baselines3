# IRPO

> **Status:** the public API below is the supported interface once the SB3-native
> IRPO engine is merged. It is not executable in the current `irpo` branch yet.

Install this fork in place of the upstream package:

```bash
pip install --upgrade \
  "stable-baselines3 @ git+https://github.com/Mgineer117/stable-baselines3.git@irpo"
```

Create a Gymnasium environment as usual, select an SB3 policy, and choose one
intrinsic-reward provider:

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
    target_kl=0.01,
    verbose=1,
)
model.learn(100_000_000)
model.save("irpo_drnd_pacman")
```

Use `"MlpPolicy"` for vector observations and `"CnnPolicy"` for image
observations. `num_options` is the number of IRPO subpolicies; each IRPO
iteration performs `num_subpolicy_updates` differentiable updates per option.

## Intrinsic reward choices

- `random`: fixed random reward functions. No pretraining.
- `drnd`: discounted random-network-distillation reward. No pretraining.
- `lirpg`: learned intrinsic-reward network trained by the IRPO outer objective.
  No pretraining.
- `allo`: frozen ALLO encoder. Pretrain it first, then pass its checkpoint:

```python
model = IRPO(
    "CnnPolicy",
    env,
    intrinsic_reward="allo",
    allo_encoder_path="allo_encoder.pt",
)
```

`allo_encoder_path` is required only for `intrinsic_reward="allo"`. The model
checkpoint will contain the selected provider and its state, so `IRPO.load()`
can resume training without a separate reward checkpoint.
