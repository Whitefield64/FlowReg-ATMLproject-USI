"""Policy building blocks shared by MiniGrid baseline and FlowReg agents."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MiniGridMLPFeatureExtractor(BaseFeaturesExtractor):
    """Learned state embedder used as MiniGrid `h_theta` for FlowReg."""

    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim)
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("MiniGridMLPFeatureExtractor expects a Box observation space")
        input_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations.float())


def build_policy_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Translate YAML policy config into SB3 `policy_kwargs`."""
    feature_config = dict(config.get("feature_extractor", {}))
    if not feature_config:
        return {}

    name = str(feature_config.get("name", "minigrid_mlp"))
    if name != "minigrid_mlp":
        raise ValueError(f"Unsupported feature extractor: {name}")

    return {
        "features_extractor_class": MiniGridMLPFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": int(feature_config.get("features_dim", 64)),
            "hidden_dim": int(feature_config.get("hidden_dim", 128)),
        },
    }
