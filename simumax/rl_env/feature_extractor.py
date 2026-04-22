"""Custom feature extractor for the PipelineSchedulingEnv Dict observation.

Ported from ``rlpp/deep_rl/feature_extractor.py`` with the ``k_f`` key
renamed to ``seq_lens`` to match the new obs layout.
"""

from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class PipelineFeatureExtractor(BaseFeaturesExtractor):
    """Per-key MLP encoder, concatenated into a single feature vector."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 64,
    ) -> None:
        n_keys = len(observation_space.spaces)
        super().__init__(observation_space, features_dim=features_dim * n_keys)

        extractors: dict[str, nn.Module] = {}
        for key, subspace in observation_space.spaces.items():
            input_dim: int = get_flattened_obs_dim(subspace)

            if key == "task_state":
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_dim, features_dim * 2),
                    nn.ReLU(),
                    nn.Linear(features_dim * 2, features_dim),
                    nn.ReLU(),
                )
            else:
                # gpu_status and seq_lens: small per-GPU / per-microbatch
                # vectors; single linear + ReLU is enough.
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_dim, features_dim),
                    nn.ReLU(),
                )

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        encoded: list[th.Tensor] = []
        for key, extractor in self.extractors.items():
            encoded.append(extractor(observations[key]))
        return th.cat(encoded, dim=1)
