"""Adapter exposing a trained MaskablePPO checkpoint as an :class:`Agent`.

Matches the duck-typed ``reset() / act(obs, mask)`` contract so a PPO
policy slots into :func:`simumax.rl.eval.evaluate` alongside the
static-schedule baselines.

The adapter is deliberately thin: it loads a ``sb3_contrib.MaskablePPO``
checkpoint via ``MaskablePPO.load`` and delegates each :meth:`act` call
to ``model.predict`` with the env-supplied action mask. Training code
(see :mod:`simumax.rl.train`) writes checkpoints that are directly
consumable here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from numpy.typing import NDArray


class PPOAgent:
    """Thin wrapper around a ``MaskablePPO`` checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.zip`` produced by Stable-Baselines3 / sb3-contrib.
    deterministic:
        If ``True`` (default), use the argmax action at eval time; if
        ``False``, sample from the policy distribution.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        *,
        deterministic: bool = True,
    ) -> None:
        # Import lazily so the rest of the agents package doesn't pull
        # in stable-baselines3 unless a PPO agent is actually used.
        from sb3_contrib import MaskablePPO

        self._checkpoint_path = str(checkpoint_path)
        self._model = MaskablePPO.load(self._checkpoint_path)
        self._deterministic = deterministic

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path

    def reset(self) -> None:
        # MaskablePPO with a feed-forward policy is stateless between
        # episodes. If we later ship a recurrent policy, reset its
        # hidden state here.
        return None

    def act(
        self,
        obs: dict[str, NDArray[Any]],
        action_mask: NDArray[Any],
    ) -> int:
        action, _state = self._model.predict(
            obs,
            action_masks=action_mask,
            deterministic=self._deterministic,
        )
        return int(action)
