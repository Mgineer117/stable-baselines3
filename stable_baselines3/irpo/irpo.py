"""SB3 integration point for IRPO.

IRPO's differentiable option-rollout engine is being ported from the research
implementation. Do not silently substitute PPO: that would report a different
algorithm under the IRPO name.
"""

from __future__ import annotations

from typing import Any

from stable_baselines3.irpo.intrinsic import IntrinsicRewardConfig


class IRPO:
    """Reserved public API for the SB3-native IRPO implementation."""

    def __init__(
        self,
        policy: str,
        env: Any,
        *,
        intrinsic_reward: str = "random",
        allo_encoder_path: str | None = None,
        num_options: int = 3,
        num_subpolicy_updates: int = 5,
        **kwargs: Any,
    ) -> None:
        if num_options < 1:
            raise ValueError("num_options must be positive")
        if num_subpolicy_updates < 2:
            raise ValueError("num_subpolicy_updates must be at least 2")
        self.policy_name = policy
        self.env = env
        self.intrinsic = IntrinsicRewardConfig(intrinsic_reward, allo_encoder_path)
        self.num_options = num_options
        self.num_subpolicy_updates = num_subpolicy_updates
        self.kwargs = kwargs
        raise NotImplementedError(
            "The IRPO package layout is in place, but the SB3-native differentiable "
            "option-rollout engine has not yet been ported."
        )
