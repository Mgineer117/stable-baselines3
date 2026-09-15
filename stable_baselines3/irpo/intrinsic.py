"""IRPO intrinsic-reward choices.

The providers live here because they are IRPO internals, not general SB3
algorithms. ALLO requires an explicit pretrained encoder checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IntrinsicReward = Literal["random", "allo", "lirpg", "drnd"]


@dataclass(frozen=True)
class IntrinsicRewardConfig:
    kind: IntrinsicReward
    allo_encoder_path: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "allo" and not self.allo_encoder_path:
            raise ValueError("ALLO requires allo_encoder_path from explicit pretraining")
        if self.kind != "allo" and self.allo_encoder_path:
            raise ValueError("allo_encoder_path is valid only with intrinsic_reward='allo'")
